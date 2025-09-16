"""
Demonstrates how to use the StreamMat library after converting data via the CLI.
"""
import asyncio
import numpy as np
import tiledb
import os
import shutil
import subprocess
from pathlib import Path
from loguru import logger

# Correct imports for package structure
from streammat.core import StreamMatrix
from streammat.logging_config import setup_logging


async def run_example():
    """Main async function to run the example."""
    setup_logging()

    # --- 1. Setup: Create a dummy Matrix Market file and define TileDB URI ---
    tiledb_array_uri = "example_matrix_tiledb"
    mm_filepath = Path("example_matrix.mtx")

    # Clean up previous runs
    if os.path.exists(tiledb_array_uri):
        logger.info(f"Removing existing directory: {tiledb_array_uri}")
        shutil.rmtree(tiledb_array_uri)
    if mm_filepath.exists():
        os.remove(mm_filepath)

    logger.info(f"Creating dummy data file: '{mm_filepath}'")
    mm_content = """%%MatrixMarket matrix coordinate real general
% A simple 4x4 sparse matrix
4 4 5
1 1 1.1
2 2 2.2
3 3 3.3
4 4 4.4
1 4 5.5
"""
    mm_filepath.write_text(mm_content)

    # --- 2. Convert the Matrix Market file using the CLI tool ---
    logger.info("--- Converting data using the 'streammat-convert' CLI tool ---")
    # This simulates running the command from the terminal.
    # Ensure the package is installed in editable mode (`pip install -e .`)
    # for the `streammat-convert` script to be available.
    cli_command = [
        "streammat-convert",
        str(mm_filepath),
        tiledb_array_uri,
        "--dtype", "float64",
        "--overwrite"  # Safe to use since we just cleaned it up
    ]
    logger.info(f"Running command: {' '.join(cli_command)}")

    # We run the conversion as a subprocess to demonstrate the CLI usage.
    result = subprocess.run(cli_command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Conversion script failed!")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        return
    logger.info(f"Conversion script output:\n{result.stdout}")

    # --- 3. Use the StreamMatrix class on the converted data ---
    if not os.path.exists(tiledb_array_uri):
        logger.error(f"TileDB array '{tiledb_array_uri}' was not created. Aborting.")
        return

    logger.info("--- Using the StreamMatrix class ---")
    ctx = tiledb.Ctx()
    matrix = StreamMatrix(uri=tiledb_array_uri, ctx=ctx)

    logger.info(f"Loaded matrix from '{matrix.uri}'")
    logger.info(f"Config: {matrix.config}")

    # --- 4. Perform matvec operation ---
    logger.info("--- Performing matvec (A * x) ---")
    input_vector_x = np.array([1, 2, 3, 4], dtype=np.float64)
    logger.info(f"Input vector x: {input_vector_x}")

    # Expected result: [23.1, 4.4, 9.9, 17.6]
    result_y = await matrix.matvec(input_vector_x)
    logger.info(f"Result vector y: {result_y}")
    np.testing.assert_allclose(result_y, [23.1, 4.4, 9.9, 17.6])
    logger.success("matvec result is correct.")

    # --- 5. Perform rmatvec operation ---
    logger.info("--- Performing rmatvec (A.T * x) ---")
    input_vector_y = np.array([10, 20, 30, 40], dtype=np.float64)
    logger.info(f"Input vector y: {input_vector_y}")

    # Expected result: [11.0, 44.0, 99.0, 231.0]
    result_z = await matrix.rmatvec(input_vector_y)
    logger.info(f"Result vector z: {result_z}")
    np.testing.assert_allclose(result_z, [11.0, 44.0, 99.0, 231.0])
    logger.success("rmatvec result is correct.")

    # --- 6. Clean up ---
    logger.warning("Cleaning up generated artifacts.")
    shutil.rmtree(tiledb_array_uri)
    os.remove(mm_filepath)
    logger.success("Cleanup complete.")


def main():
    asyncio.run(run_example())

if __name__ == "__main__":
    main()
