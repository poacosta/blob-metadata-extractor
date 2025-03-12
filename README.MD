# Blob Metadata Extractor 🗄️: A Developer's Tale

Ever found yourself with 400k+ files and a database that needs to know about them? Welcome to my world - and the
solution that emerged from it.

## The Problem Space 🤔

Let's be honest: file metadata extraction sits in that awkward zone between "too boring to be glamorous" and "too
critical to ignore." When faced with migrating vast file collections into Rails Active Storage (or any structured
system), the gap between our filesystem reality and database expectations becomes painfully apparent.

This tool bridges that chasm, automating metadata extraction with an emphasis on robustness, performance, and clean code
practices that won't make your future self curse your name.

## Tool Philosophy & Design 🧠

Rather than just throwing together a quick script, I approached this with battle-tested engineering principles:

- **Parallel By Default**: Because life's too short to process files sequentially
- **Explicit Error Boundaries**: Failing gracefully when things go sideways (and with filesystems, they will)
- **Memory Conscious**: Batch processing keeps the RAM footprint reasonable
- **Clean Code Practices**: Specific exception handling, proper logging patterns, and linting compliance
- **Type Safety**: Comprehensive type annotations throughout for better maintainability

## Getting Started 🚀

### Installation

```bash
# Clone the repository
git clone https://github.com/poacosta/blob-metadata-extractor.git
cd blob-metadata-extractor

# Install dependencies
pip install -r requirements.txt

# Platform-specific libmagic installation
# For macOS
brew install libmagic

# For Ubuntu/Debian
apt-get install libmagic-dev

# For Windows
# See python-magic-bin documentation
```

### Core Dependencies

- **pandas**: For efficient data manipulation
- **tqdm**: For progress tracking that doesn't make you question if the script died
- **python-magic**: For content type inspection that goes beyond "guess by extension"
- **Pillow**: For extracting image dimensions and metadata (optional)

## Usage Examples 💻

### Process a Directory Tree

```bash
python blob_metadata_extractor.py \
  --input-path /path/to/files \
  --output-csv output.csv \
  --key-prefix "storage/" \
  --workers 8
```

### Process Files Listed in a CSV

```bash
python blob_metadata_extractor.py \
  --input-csv paths.csv \
  --output-csv output.csv \
  --start-id 1001 \
  --service-name "s3"
```

### Handle Relative Paths with Base Directory

```bash
python blob_metadata_extractor.py \
  --input-csv relative_paths.csv \
  --root-path "C:\Users\username\Documents\Datasets\logs" \
  --output-csv output.csv \
  --key-prefix "storage/"
```

### Set Custom Creation Date

```bash
python blob_metadata_extractor.py \
  --input-path /path/to/files \
  --output-csv output.csv \
  --created-at "2025-04-15 09:30:00"
```

### Skip Image Dimension Analysis

```bash
python blob_metadata_extractor.py \
  --input-path /path/to/files \
  --output-csv output.csv \
  --skip-image-analysis
```

### Full Command Reference

```
usage: blob_metadata_extractor.py [-h] (--input-path INPUT_PATH | --input-csv INPUT_CSV) --output-csv OUTPUT_CSV
                                 [--root-path ROOT_PATH] [--key-prefix KEY_PREFIX] [--service-name SERVICE_NAME]
                                 [--created-at CREATED_AT] [--workers WORKERS] [--batch-size BATCH_SIZE] 
                                 [--start-id START_ID] [--skip-image-analysis] [--verbose]

Extract metadata from files and generate a CSV file matching Active Storage blobs schema.

options:
  -h, --help            show this help message and exit
  --input-path INPUT_PATH
                        Root directory path to scan for files
  --input-csv INPUT_CSV
                        CSV file containing file paths (single column)
  --output-csv OUTPUT_CSV
                        Path to output CSV file
  --root-path ROOT_PATH
                        Root path to prepend to relative paths in the CSV file
  --key-prefix KEY_PREFIX
                        Optional prefix to add to the "key" column values
  --service-name SERVICE_NAME
                        Value for the service_name column (default: local)
  --created-at CREATED_AT
                        Value for the created_at column (default: current date/time, format: YYYY-MM-DD HH:MM:SS)
  --workers WORKERS     Number of worker processes to use (default: available CPU cores)
  --batch-size BATCH_SIZE
                        Number of files to process in each batch (default: 1000)
  --start-id START_ID   Starting ID for the id column (default: 1)
  --skip-image-analysis Skip extraction of image dimensions and analysis
  --verbose             Enable verbose logging
```

## Engineering Insights 🔍

### Code Quality Focus

You might not expect a utility script to emphasize clean code, but there's a method to this madness:

- **Proper Logging**: Using `%` formatting instead of f-strings for optimal performance
- **Specific Exceptions**: Targeting `FileNotFoundError` and `IOError` before falling back to generic catches
- **Cross-Platform Path Handling**: Normalizing slashes for Windows/Unix compatibility
- **Memory Efficiency**: Streaming large directories with generators
- **Type Annotations**: Comprehensive typing throughout for better IDE support and static analysis
- **Parameter Validation**: Explicit handling of optional parameters with proper defaults

## Limitations

### Current Constraints

- **Memory Footprint**: Still loads all results before CSV creation
- **Limited Metadata**: Focused on Active Storage compatibility, not richer media metadata
- **Single Machine**: No distributed processing capability (yet)

## License 📄

This project is licensed under the MIT License.

## Final Thoughts 💭

Building this tool reminded me why "boring" utilities often reveal the most interesting engineering challenges. What
started as "just extract some metadata" evolved into a playground for parallel processing, error handling patterns, and
filesystem quirks.

The next time you're tasked with a seemingly mundane file operation at scale, remember: there's elegant engineering to
be found in even the most utilitarian corners of our craft.

---

*"Good code is like a good joke - it needs no explanation."*