#!/usr/bin/env python3
"""
Blob Metadata Extractor

A command-line tool that extracts metadata from files (primarily images) and generates
a CSV file with information matching the Active Storage blobs schema.

This tool recursively scans directories or processes file paths from a CSV file,
extracts metadata like content type and size, and generates a structured CSV output
that can be used for database imports or file inventory purposes.
"""

import os
import sys
import json
import argparse
import logging
import mimetypes
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List, Dict, Union, Optional, Generator, Tuple

import magic  # python-magic for more accurate content type detection
import pandas as pd
from tqdm import tqdm

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('blob-metadata-extractor')


def extract_image_metadata(file_path: str) -> Dict[str, Union[bool, int]]:
    """
    Extract metadata from an image file.

    Args:
        file_path: Path to the image file

    Returns:
        Dictionary containing image metadata
    """
    if not HAS_PIL:
        return {"identified": True, "analyzed": True}

    try:
        with Image.open(file_path) as img:
            width, height = img.size
            return {
                "identified": True,
                "width": width,
                "height": height,
                "analyzed": True
            }
    except Exception as e:
        logger.warning("Failed to extract image dimensions from %s: %s", file_path, e)
        return {"identified": True, "analyzed": True}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract metadata from files and generate a CSV file matching '
                    'Active Storage blobs schema.'
    )

    # Input options - mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-path',
        help='Root directory path to scan for files'
    )
    input_group.add_argument(
        '--input-csv',
        help='CSV file containing file paths (single column)'
    )

    # Optional root path for CSV file paths
    parser.add_argument(
        '--root-path',
        help='Root path to prepend to relative paths in the CSV file'
    )

    # Output options
    parser.add_argument(
        '--output-csv',
        required=True,
        help='Path to output CSV file'
    )

    # Additional options
    parser.add_argument(
        '--key-prefix',
        default='',
        help='Optional prefix to add to the "key" column values'
    )
    parser.add_argument(
        '--service-name',
        default='local',
        help='Value for the service_name column (default: local)'
    )
    parser.add_argument(
        '--created-at',
        help='Value for the created_at column (default: current date/time, format: YYYY-MM-DD HH:MM:SS)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=os.cpu_count(),
        help=f'Number of worker processes to use (default: {os.cpu_count()})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of files to process in each batch (default: 1000)'
    )
    parser.add_argument(
        '--start-id',
        type=int,
        default=1,
        help='Starting ID for the id column (default: 1)'
    )
    parser.add_argument(
        '--skip-image-analysis',
        action='store_true',
        help='Skip extraction of image dimensions and analysis'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def get_file_paths(input_path: str) -> Generator[str, None, None]:
    """
    Generate a list of all file paths from the given directory.

    Args:
        input_path: Root directory to scan

    Returns:
        Generator yielding absolute file paths
    """
    logger.info("Scanning directory: %s", input_path)

    for root, _, files in os.walk(input_path):
        for filename in files:
            if filename.startswith('.'):
                continue

            yield os.path.abspath(os.path.join(root, filename))


def read_paths_from_csv(csv_path: str, root_path: Optional[str] = None) -> List[str]:
    """
    Read file paths from a CSV file.

    Args:
        csv_path: Path to the CSV file
        root_path: Optional root path to prepend to relative paths

    Returns:
        List of file paths
    """
    logger.info("Reading file paths from CSV: %s", csv_path)

    try:
        df = pd.read_csv(csv_path)
        path_column = df.columns[0]
        paths = df[path_column].tolist()

        if root_path:
            logger.info("Prepending root path: %s", root_path)
            paths = [
                os.path.join(
                    root_path,
                    path.replace('\\', os.sep).replace('/', os.sep)
                ) for path in paths
            ]

        return paths
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty: %s", csv_path)
        sys.exit(1)
    except pd.errors.ParserError:
        logger.error("Error parsing CSV file: %s", csv_path)
        sys.exit(1)
    except FileNotFoundError:
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)
    except IOError as e:
        logger.error("I/O error reading CSV file: %s", e)
        sys.exit(1)
    except Exception as e:
        # Fallback for unexpected errors
        logger.error("Unexpected error reading CSV file: %s", e)
        sys.exit(1)


def get_content_type(file_path: str) -> str:
    """
    Determine the content type (MIME type) of a file.

    Args:
        file_path: Path to the file

    Returns:
        Content type string
    """
    try:
        # First try using python-magic for more accurate detection
        content_type = magic.from_file(file_path, mime=True)

        # Fall back to mimetypes if magic fails
        if not content_type:
            content_type, _ = mimetypes.guess_type(file_path)

        return content_type or 'application/octet-stream'
    except (IOError, OSError) as e:
        logger.warning("I/O error determining content type for %s: %s", file_path, e)
        return 'application/octet-stream'
    except Exception as e:
        logger.warning("Error determining content type for %s: %s", file_path, e)
        return 'application/octet-stream'


def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes
    """
    try:
        return os.path.getsize(file_path)
    except (IOError, OSError) as e:
        logger.warning("I/O error getting file size for %s: %s", file_path, e)
        return 0
    except Exception as e:
        logger.warning("Error getting file size for %s: %s", file_path, e)
        return 0


def extract_metadata(
        file_path: str,
        key_prefix: str,
        service_name: str,
        base_path: Optional[str] = None,
        created_at: Optional[str] = None,
        skip_image_analysis: bool = False
) -> Dict[str, Union[str, int]]:
    """
    Extract metadata from a file.

    Args:
        file_path: Path to the file
        key_prefix: Prefix to add to the key
        service_name: Value for service_name column
        base_path: Base path for calculating relative paths
        created_at: Value for created_at column
        skip_image_analysis: Whether to skip image metadata extraction

    Returns:
        Dictionary containing extracted metadata
    """
    filename = os.path.basename(file_path)

    if base_path:
        try:
            relative_path = os.path.relpath(file_path, base_path)
        except ValueError:
            # Handle case where paths are on different drives (Windows)
            logger.warning("Could not calculate relative path from %s to %s", base_path, file_path)
            relative_path = filename
    else:
        relative_path = filename

    relative_path = relative_path.replace('\\', '/')
    key = f"{key_prefix}{relative_path}" if key_prefix else relative_path

    content_type = get_content_type(file_path)
    byte_size = get_file_size(file_path)

    metadata_dict = {}
    if not skip_image_analysis and content_type and content_type.startswith('image/'):
        metadata_dict = extract_image_metadata(file_path)

    metadata = json.dumps(metadata_dict)

    created_at_value = created_at or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    checksum = ""

    return {
        'key': key,
        'filename': filename,
        'content_type': content_type,
        'metadata': metadata,
        'service_name': service_name,
        'byte_size': byte_size,
        'checksum': checksum,
        'created_at': created_at_value
    }


def process_file_batch(
        batch: List[Tuple[int, str, str, str, str, str, bool]]
) -> List[Dict[str, Union[str, int]]]:
    """
    Process a batch of files to extract metadata.

    Args:
        batch: List of tuples containing (id, file_path, key_prefix, service_name,
               base_path, created_at, skip_image_analysis)

    Returns:
        List of dictionaries containing file metadata
    """
    results = []

    for id_num, file_path, key_prefix, service_name, base_path, created_at, skip_image_analysis in batch:
        try:
            metadata = extract_metadata(
                file_path,
                key_prefix,
                service_name,
                base_path,
                created_at,
                skip_image_analysis
            )
            metadata['id'] = id_num
            results.append(metadata)
        except FileNotFoundError:
            logger.error("File not found: %s", file_path)
        except (IOError, OSError) as e:
            logger.error("I/O error processing file %s: %s", file_path, e)
        except Exception as e:
            # Fallback for unexpected errors
            logger.error("Error processing file %s: %s", file_path, e)

    return results


def main() -> None:
    """Main function to run the blob metadata extractor."""
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.skip_image_analysis and not HAS_PIL:
        logger.warning("PIL/Pillow is not installed. Image dimension extraction will be skipped.")
        logger.warning("Install with: pip install Pillow")

    created_at = args.created_at or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not args.created_at:
        logger.info("Using current date/time for created_at: %s", created_at)

    file_paths = []
    base_path = None

    if args.input_path:
        file_paths = list(get_file_paths(args.input_path))
        base_path = args.input_path
    elif args.input_csv:
        file_paths = read_paths_from_csv(args.input_csv, args.root_path)
        base_path = args.root_path

    total_files = len(file_paths)
    logger.info("Found %s files to process", total_files)

    if total_files == 0:
        logger.warning("No files found to process")
        sys.exit(0)

    batches = []
    current_id = args.start_id

    for i in range(0, total_files, args.batch_size):
        batch = []
        for j in range(i, min(i + args.batch_size, total_files)):
            batch.append((
                current_id,
                file_paths[j],
                args.key_prefix,
                args.service_name,
                base_path,
                created_at,
                args.skip_image_analysis
            ))
            current_id += 1
        batches.append(batch)

    logger.info("Processing files using %s workers", args.workers)
    all_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file_batch, batch) for batch in batches]

        for future in tqdm(futures, total=len(futures), desc="Processing batches"):
            batch_results = future.result()
            all_results.extend(batch_results)

    df = pd.DataFrame(all_results)

    column_order = [
        'id', 'key', 'filename', 'content_type', 'metadata',
        'service_name', 'byte_size', 'checksum', 'created_at'
    ]
    df = df[column_order]

    logger.info("Saving results to %s", args.output_csv)
    df.to_csv(args.output_csv, index=False)

    logger.info("Processed %s files", len(all_results))
    logger.info("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)
