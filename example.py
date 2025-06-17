"""
Demonstrates how to use the StreamMat library and convert data.
"""
import asyncio
import os
import shutil

import numpy as np
import tiledb
from loguru import logger

from conversion import convert_matrix_market_to_tiledb
from core import StreamMatrix
from logging_config import setup_logging
from models import DataType


async def main():
    """Main async function to run the example."""
    # Setup logging for the script
    setup_logging()

    # --- 1. Setup: Create a dummy Matrix Market file and a TileDB URI ---
    tiledb_array_uri = "example_matrix_tiledb"
    mm_filepath = "example_matrix.mtx"

    # Clean up previous runs
    if os.path.exists(tiledb_array_uri):
        logger.info(f"Removing existing directory: {tiledb_array_uri}")
        shutil.rmtree(tiledb_array_uri)
    if os.path.exists(mm_filepath):
        os.remove(mm_filepath)

    # Create a simple 4x4 matrix in Matrix Market format
    mm_content = """%%MatrixMarket matrix coordinate real general
% A simple 4x4 sparse matrix
4 4 5
1 1 1.1
2 2 2.2
3 3 3.3
4 4 4.4
1 4 5.5
"""
    with open(mm_filepath, "w") as f:
        f.write(mm_content)

    # --- 2. Convert the Matrix Market file to a TileDB array ---
    convert_matrix_market_to_tiledb(
        mm_filepath=mm_filepath,
        tiledb_uri=tiledb_array_uri,
        target_dtype=DataType.FLOAT64
    )

    # --- 3. Use the StreamMatrix class ---
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

    # --- 5. Perform rmatvec operation ---
    logger.info("--- Performing rmatvec (A.T * x) ---")
    input_vector_y = np.array([10, 20, 30, 40], dtype=np.float64)
    logger.info(f"Input vector y: {input_vector_y}")

    # Expected result: [11.0, 44.0, 99.0, 231.0]
    result_z = await matrix.rmatvec(input_vector_y)
    logger.info(f"Result vector z: {result_z}")

    # --- 6. Clean up ---
    logger.warning("Cleaning up generated artifacts.")
    shutil.rmtree(tiledb_array_uri)
    os.remove(mm_filepath)
    logger.success("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
