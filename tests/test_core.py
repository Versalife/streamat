import os
import shutil
from collections.abc import Generator

import numpy as np
import pytest
import tiledb

from streammat.core import ATTR_VALUE_NAME, DIM_COL_NAME, DIM_ROW_NAME, StreamMatrix
from streammat.models import DataType, ErrorCode, StreamMatConfig, StreamMatException

TEST_MATRIX_URI = "test_core_matrix_tiledb"


@pytest.fixture(scope="module")
def setup_and_teardown_tiledb() -> Generator[None, None, None]:
    """Fixture to create and clean up a test TileDB array for core tests."""
    if os.path.exists(TEST_MATRIX_URI):
        shutil.rmtree(TEST_MATRIX_URI)

    rows, cols = 4, 4
    domain = tiledb.Domain(
        tiledb.Dim(name=DIM_ROW_NAME, domain=(0, rows - 1), tile=4, dtype=np.uint64),
        tiledb.Dim(name=DIM_COL_NAME, domain=(0, cols - 1), tile=4, dtype=np.uint64),
    )
    schema = tiledb.ArraySchema(
        domain=domain,
        sparse=True,
        attrs=[tiledb.Attr(name=ATTR_VALUE_NAME, dtype=np.float64)],
    )
    tiledb.Array.create(TEST_MATRIX_URI, schema)

    row_coords = np.array([0, 1, 2, 3, 0])
    col_coords = np.array([0, 1, 2, 3, 3])
    values = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    nnz = len(values)

    with tiledb.open(TEST_MATRIX_URI, "w") as A:
        A[row_coords, col_coords] = values
        config = StreamMatConfig(rows=rows, cols=cols, nnz=nnz, dtype=DataType.FLOAT64, version=1)
        for field, value in config.model_dump(by_alias=True).items():
            A.meta[field] = value.value if isinstance(value, DataType) else value

    yield

    if os.path.exists(TEST_MATRIX_URI):
        shutil.rmtree(TEST_MATRIX_URI)


@pytest.fixture
def sparse_matrix_numpy() -> np.ndarray:
    """Provides the numpy equivalent of the test matrix."""
    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 0] = 1.1
    A[1, 1] = 2.2
    A[2, 2] = 3.3
    A[3, 3] = 4.4
    A[0, 3] = 5.5
    return A


@pytest.mark.asyncio
async def test_matvec(
    setup_and_teardown_tiledb: None, sparse_matrix_numpy: np.ndarray
) -> None:
    ctx = tiledb.Ctx()
    matrix = StreamMatrix(uri=TEST_MATRIX_URI, ctx=ctx)
    vector = np.array([1, 2, 3, 4], dtype=np.float64)

    result_streammat = await matrix.matvec(vector)
    result_numpy = sparse_matrix_numpy.dot(vector)

    np.testing.assert_allclose(result_streammat, result_numpy)


@pytest.mark.asyncio
async def test_rmatvec(
    setup_and_teardown_tiledb: None, sparse_matrix_numpy: np.ndarray
) -> None:
    ctx = tiledb.Ctx()
    matrix = StreamMatrix(uri=TEST_MATRIX_URI, ctx=ctx)
    vector = np.array([10, 20, 30, 40], dtype=np.float64)

    result_streammat = await matrix.rmatvec(vector)
    result_numpy = sparse_matrix_numpy.T.dot(vector)

    np.testing.assert_allclose(result_streammat, result_numpy)


@pytest.mark.asyncio
async def test_matvec_chunked(
    setup_and_teardown_tiledb: None, sparse_matrix_numpy: np.ndarray
) -> None:
    ctx = tiledb.Ctx()
    matrix = StreamMatrix(uri=TEST_MATRIX_URI, ctx=ctx)
    vector = np.array([1, 2, 3, 4], dtype=np.float64)

    result_streammat = await matrix.matvec(vector, chunk_size=2)
    result_numpy = sparse_matrix_numpy.dot(vector)

    np.testing.assert_allclose(result_streammat, result_numpy)


@pytest.mark.asyncio
async def test_rmatvec_chunked(
    setup_and_teardown_tiledb: None, sparse_matrix_numpy: np.ndarray
) -> None:
    ctx = tiledb.Ctx()
    matrix = StreamMatrix(uri=TEST_MATRIX_URI, ctx=ctx)
    vector = np.array([10, 20, 30, 40], dtype=np.float64)

    result_streammat = await matrix.rmatvec(vector, chunk_size=2)
    result_numpy = sparse_matrix_numpy.T.dot(vector)

    np.testing.assert_allclose(result_streammat, result_numpy)


@pytest.mark.asyncio
async def test_matvec_dimension_mismatch(setup_and_teardown_tiledb: None) -> None:
    ctx = tiledb.Ctx()
    matrix = StreamMatrix(uri=TEST_MATRIX_URI, ctx=ctx)
    vector = np.array([1, 2, 3], dtype=np.float64)  # Incorrect shape

    with pytest.raises(StreamMatException) as excinfo:
        await matrix.matvec(vector)
    assert excinfo.value.code == ErrorCode.DIMENSION_MISMATCH


@pytest.mark.asyncio
async def test_rmatvec_dimension_mismatch(setup_and_teardown_tiledb: None) -> None:
    ctx = tiledb.Ctx()
    matrix = StreamMatrix(uri=TEST_MATRIX_URI, ctx=ctx)
    vector = np.array([1, 2, 3, 4, 5], dtype=np.float64)  # Incorrect shape

    with pytest.raises(StreamMatException) as excinfo:
        await matrix.rmatvec(vector)
    assert excinfo.value.code == ErrorCode.DIMENSION_MISMATCH
