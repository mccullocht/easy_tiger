# Mongodump Archive Reader

A command-line tool to read and inspect MongoDB dump archives created by `mongodump --archive`.

## Usage

### Basic Usage

Read the first 5 documents from each collection in an archive:

```bash
cargo run -p et -- mongodump read --archive /path/to/dump.archive
```

### Options

- `--archive, -a <PATH>`: Path to the mongodump archive file (required)
- `--count, -n <NUMBER>`: Number of documents to display per collection (default: 5)
- `--database, -d <NAME>`: Filter by database name (optional)
- `--collection, -c <NAME>`: Filter by collection name (optional)

### Examples

Show 10 documents from each collection:
```bash
cargo run -p et -- mongodump read --archive dump.archive --count 10
```

Show documents only from a specific database:
```bash
cargo run -p et -- mongodump read --archive dump.archive --database mydb
```

Show documents from a specific collection:
```bash
cargo run -p et -- mongodump read --archive dump.archive --database mydb --collection users
```

## Archive Format

The tool reads mongodump archives according to the format specification in `spec.md`. The archive contains:

1. **Magic number**: `0x6de29981` (little-endian)
2. **Header**: BSON document with archive metadata
3. **Collection metadata**: One or more BSON documents describing each collection
4. **Terminator**: `0xffffffff`
5. **Namespace segments**: Interleaved data from multiple collections

Each namespace segment contains:
- **Namespace header**: BSON document with db/collection name and EOF flag
- **Documents**: One or more BSON documents from the collection
- **Terminator**: `0xffffffff`

## Implementation Details

The reader:
- Validates the magic number at the start of the archive
- Parses BSON documents using the `bson` crate
- Handles interleaved collection data
- Supports filtering by database and collection name
- Limits output to a configurable number of documents per collection

