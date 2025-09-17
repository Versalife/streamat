import os
import shutil
import subprocess
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import tiledb

from streammat.core import ATTR_VALUE_NAME, DIM_COL_NAME, DIM_ROW_NAME
from streammat.models import DataType, StreamMatConfig

MM_CONTENT = """%%MatrixMarket matrix coordinate real general
% A simple 4x4 sparse matrix
4 4 5
1 1 1.1
2 2 2.2
3 3 3.3
4 4 4.4
1 4 5.5
"""


@pytest.fixture
def setup_test_files() -> Generator[tuple[Path, str], None, None]:
    """Create a dummy matrix market file for testing."""
    mm_filepath = Path("test_matrix.mtx")
    mm_filepath.write_text(MM_CONTENT)
    output_uri = "test_conversion_output"

    yield mm_filepath, output_uri

    # Teardown
    if mm_filepath.exists():
        os.remove(mm_filepath)
    if os.path.exists(output_uri):
        shutil.rmtree(output_uri)


def test_conversion_cli_success(setup_test_files: tuple[Path, str]) -> None:
    """Test that the streammat-convert CLI runs successfully."""
    mm_filepath, output_uri = setup_test_files

    cli_command = [
        "streammat-convert",
        str(mm_filepath),
        output_uri,
        "--dtype",
        "float32",
        "--overwrite",
    ]

    result = subprocess.run(cli_command, capture_output=True, text=True)

    assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"
    assert Path(output_uri).exists()

    # Verify the metadata of the created array
    with tiledb.open(output_uri, "r") as A:
        meta = {k: v for k, v in A.meta.items()}
        config = StreamMatConfig.model_validate(meta)
        assert config.rows == 4
        assert config.cols == 4
        assert config.nnz == 5
        assert config.dtype == DataType.FLOAT32

    # Verify the contents of the created array
    with tiledb.open(output_uri, "r") as A:
        data = A[:]
        rows = data[DIM_ROW_NAME]
        cols = data[DIM_COL_NAME]
        vals = data[ATTR_VALUE_NAME]

        # Sort by row, then col for consistent comparison
        sort_idx = np.lexsort((cols, rows))
        rows, cols, vals = rows[sort_idx], cols[sort_idx], vals[sort_idx]

        expected_rows = np.array([0, 0, 1, 2, 3])
        expected_cols = np.array([0, 3, 1, 2, 3])
        expected_vals = np.array([1.1, 5.5, 2.2, 3.3, 4.4], dtype=np.float32)

        np.testing.assert_array_equal(rows, expected_rows)
        np.testing.assert_array_equal(cols, expected_cols)
        np.testing.assert_allclose(vals, expected_vals, rtol=1e-5)


def test_conversion_cli_no_overwrite(setup_test_files: tuple[Path, str]) -> None:
    """Test that the CLI fails if the output exists and --overwrite is not provided."""
    mm_filepath, output_uri = setup_test_files
    os.makedirs(output_uri, exist_ok=True)

    cli_command = [
        "streammat-convert",
        str(mm_filepath),
        output_uri,
    ]

    result = subprocess.run(cli_command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "already exists" in result.stderr.lower()
