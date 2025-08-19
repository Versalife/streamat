"""
Utility functions for converting data from other formats to TileDB arrays
compatible with StreamMat.
"""
import enum
import gzip
import shutil
from pathlib import Path
from typing import Generator, TextIO, Tuple

import numpy as np
import tiledb
import typer
from loguru import logger

from .core import ATTR_VALUE_NAME, DIM_COL_NAME, DIM_ROW_NAME
from .logging_config import setup_logging
from .models import DataType, ErrorCode, StreamMatConfig, StreamMatException

# --- Type Definitions ---
# A standardized internal representation for a chunk of sparse data
SparseChunk = np.ndarray  # A structured array with fields ('row', 'col', 'val')
# A generator that yields metadata first, then chunks of sparse data
StreamedMatrixGenerator = Generator[Tuple[int, int, int] | SparseChunk, None, None]

# --- CLI Setup ---
cli_app = typer.Typer(
    name="streammat-convert",
    help="A tool to convert various sparse matrix formats to a StreamMat-compatible TileDB array.",
    add_completion=False,
)


class MatrixFormat(str, enum.Enum):
    """Supported input matrix formats."""
    MATRIX_MARKET = "matrix-market"
    HARWELL_BOEING = "harwell-boeing"
    GCT = "gct"


# --- Reader Components ---

def _read_matrix_market(f: TextIO, dtype: np.dtype) -> StreamedMatrixGenerator:
    """Generator to read a Matrix Market file and yield metadata and data chunks."""
    line = f.readline()
    while line.startswith('%'):
        line = f.readline()
    rows, cols, nnz = map(int, line.split())
    yield rows, cols, nnz

    lines = []
    chunk_size = 1_000_000
    for line in f:
        lines.append(line)
        if len(lines) == chunk_size:
            data = np.empty(len(lines), dtype=[('row', np.uint64), ('col', np.uint64), ('val', dtype)])
            for i, l in enumerate(lines):
                row, col, val = l.split()
                data[i] = (int(row) - 1, int(col) - 1, val)
            yield data
            lines = []
    if lines:
        data = np.empty(len(lines), dtype=[('row', np.uint64), ('col', np.uint64), ('val', dtype)])
        for i, l in enumerate(lines):
            row, col, val = l.split()
            data[i] = (int(row) - 1, int(col) - 1, val)
        yield data


def _read_harwell_boeing(f: TextIO, dtype: np.dtype) -> StreamedMatrixGenerator:
    """Generator to read a Harwell-Boeing file using SciPy."""
    try:
        from scipy.io import hb_read
    except ImportError:
        raise StreamMatException(
            ErrorCode.MISSING_DEPENDENCY,
            "The 'scipy' library is required to read Harwell-Boeing formats. Please run: pip install scipy"
        )

    csc_matrix = hb_read(f)
    rows, cols = csc_matrix.shape
    nnz = csc_matrix.nnz
    yield rows, cols, nnz

    # Convert CSC to COO format (row, col, val) for writing
    coo_matrix = csc_matrix.tocoo()
    data = np.empty(
        nnz,
        dtype=[('row', np.uint64), ('col', np.uint64), ('val', dtype)]
    )
    data['row'] = coo_matrix.row
    data['col'] = coo_matrix.col
    data['val'] = coo_matrix.data.astype(dtype)
    yield data


def _read_gct(f: TextIO, dtype: np.dtype) -> StreamedMatrixGenerator:
    """Generator to read a GCT file and yield metadata and sparse data chunks."""
    _ = f.readline()  # Skip version line
    rows, cols = map(int, f.readline().split())
    _ = f.readline()  # Skip column headers line

    nnz = 0
    lines = []
    chunk_size = 100_000  # Number of rows to process at a time

    # First pass to count non-zero elements
    logger.info("First pass (GCT): Counting non-zero elements...")
    for row_idx, line in enumerate(f):
        parts = line.strip().split('\t')
        values = np.array(parts[2:], dtype=dtype)
        nnz += np.count_nonzero(values)

    yield rows, cols, nnz
    logger.info(f"GCT matrix has {rows}x{cols} dimensions with {nnz} non-zero entries.")

    # Second pass to yield data chunks
    f.seek(0)
    _ = f.readline()
    _ = f.readline()
    _ = f.readline()  # Reset to start of data

    logger.info("Second pass (GCT): Reading and yielding data chunks...")
    for row_idx, line in enumerate(f):
        parts = line.strip().split('\t')
        values = np.array(parts[2:], dtype=dtype)

        # Find column indices of non-zero elements
        col_indices = np.nonzero(values)[0]

        for col_idx in col_indices:
            lines.append((row_idx, col_idx, values[col_idx]))

        if len(lines) >= chunk_size:
            data = np.array(lines, dtype=[('row', np.uint64), ('col', np.uint64), ('val', dtype)])
            yield data
            lines = []

    if lines:
        data = np.array(lines, dtype=[('row', np.uint64), ('col', np.uint64), ('val', dtype)])
        yield data


# --- Writer Component ---

def _create_and_write_tiledb(
    tiledb_uri: str,
    target_dtype: DataType,
    matrix_generator: StreamedMatrixGenerator,
    ctx: tiledb.Ctx,
):
    """Creates and writes data to a TileDB array from a generator."""
    numpy_dtype = np.float64 if target_dtype == DataType.FLOAT64 else np.float32

    # The first item from the generator must be the metadata
    try:
        rows, cols, nnz = next(matrix_generator)
    except StopIteration:
        raise StreamMatException(ErrorCode.UNSUPPORTED_FORMAT, "Input file is empty or has no header.")

    logger.info(f"Matrix dimensions: {rows}x{cols}, NNZ: {nnz}")
    logger.info("Creating TileDB array schema...")

    domain = tiledb.Domain(
        tiledb.Dim(name=DIM_ROW_NAME, domain=(0, rows - 1), tile=min(rows, 4096), dtype=np.uint64),
        tiledb.Dim(name=DIM_COL_NAME, domain=(0, cols - 1), tile=min(cols, 4096), dtype=np.uint64),
    )
    schema = tiledb.ArraySchema(
        domain=domain,
        sparse=True,
        attrs=[tiledb.Attr(name=ATTR_VALUE_NAME, dtype=numpy_dtype)]
    )
    tiledb.Array.create(tiledb_uri, schema, ctx=ctx)

    logger.info("Writing data to TileDB array...")
    with tiledb.open(tiledb_uri, 'w', ctx=ctx) as A:
        for i, data_chunk in enumerate(matrix_generator):
            logger.debug(f"Writing chunk {i + 1} with {len(data_chunk)} entries...")
            A[data_chunk['row'], data_chunk['col']] = data_chunk['val']

        logger.info("Writing metadata...")
        config = StreamMatConfig(rows=rows, cols=cols, nnz=nnz, dtype=target_dtype)
        for field, value in config.model_dump(by_alias=True).items():
            A.meta[field] = value.value if isinstance(value, enum.Enum) else value


# --- Main CLI Function ---

@cli_app.command()
def convert(
    input_path: Path = typer.Argument(
        ...,
        help="""Path to the source sparse matrix file (e.g., matrix.mtx, matrix.hb, matrix.gct).
        Handles .gz automatically.""",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_uri: str = typer.Argument(
        ...,
        help="URI for the new TileDB array to be created (can be a local path or a cloud URI like 'tiledb://...')."
    ),
    matrix_format: MatrixFormat = typer.Option(
        None,
        "--format", "-f",
        help="The format of the input file. If not provided, it will be inferred from the file extension.",
        case_sensitive=False,
    ),
    target_dtype: DataType = typer.Option(
        DataType.FLOAT64,
        "--dtype",
        help="The data type for storing values in the TileDB array.",
        case_sensitive=False,
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite the output TileDB array if it already exists.",
    )
):
    """
    Converts a sparse matrix file to a StreamMat-compatible TileDB array.
    """
    setup_logging()

    if matrix_format is None:
        suffix = input_path.suffix
        if suffix == ".gz":
            suffix = input_path.with_suffix("").suffix  # Check extension before .gz

        if suffix in [".mtx", ".mm"]:
            matrix_format = MatrixFormat.MATRIX_MARKET
        elif suffix in [".hb", ".rua"]:
            matrix_format = MatrixFormat.HARWELL_BOEING
        elif suffix == ".gct":
            matrix_format = MatrixFormat.GCT
        else:
            logger.error(f"Could not infer format from file extension '{suffix}'. Please specify with --format.")
            raise typer.Exit(code=1)
        logger.info(f"Inferred input format as: {matrix_format.value}")

    if Path(output_uri).exists():
        if overwrite:
            logger.warning(f"Output URI '{output_uri}' exists. Overwriting as requested.")
            shutil.rmtree(output_uri)
        else:
            logger.error(f"Output URI '{output_uri}' already exists. Use --overwrite to replace it.")
            raise typer.Exit(code=1)

    try:
        ctx = tiledb.Ctx()
        numpy_dtype = np.float64 if target_dtype == DataType.FLOAT64 else np.float32

        opener = gzip.open if input_path.suffix == ".gz" else open

        with opener(input_path, "rt", encoding="utf-8") as f:
            if matrix_format == MatrixFormat.MATRIX_MARKET:
                generator = _read_matrix_market(f, numpy_dtype)
            elif matrix_format == MatrixFormat.HARWELL_BOEING:
                generator = _read_harwell_boeing(f, numpy_dtype)
            elif matrix_format == MatrixFormat.GCT:
                generator = _read_gct(f, numpy_dtype)
            else:
                # This case should not be reachable due to earlier checks
                raise StreamMatException(ErrorCode.UNSUPPORTED_FORMAT, f"Format '{matrix_format}' not implemented.")

            _create_and_write_tiledb(output_uri, target_dtype, generator, ctx)

        logger.success(f"Conversion complete. TileDB array created at '{output_uri}'.")

    except StreamMatException as e:
        logger.error(f"A StreamMat error occurred: {e.message} (Code: {e.code.value})")
        raise typer.Exit(code=1)
    except Exception:
        logger.exception("An unexpected error occurred during the conversion process.")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    cli_app()
