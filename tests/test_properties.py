import asyncio
import os
import shutil

import numpy as np
import tiledb
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import DrawFn, composite, floats, integers
from scipy.sparse import coo_matrix

from streammat.conversion import convert_coo_to_tiledb
from streammat.core import StreamMatrix
from streammat.models import DataType

# --- Hypothesis Strategies ---


@composite
def sparse_matrices(draw: DrawFn) -> coo_matrix:
    """A Hypothesis strategy to generate SciPy COO sparse matrices."""
    dtype = np.float64
    rows = draw(integers(min_value=1, max_value=100))
    cols = draw(integers(min_value=1, max_value=100))

    # Generate a limited number of non-zero elements
    nnz = draw(integers(min_value=0, max_value=rows * cols // 2))

    if nnz == 0:
        return coo_matrix((rows, cols), dtype=dtype)

    # Generate unique coordinates for the sparse matrix
    coords: set[tuple[int, int]] = set()
    while len(coords) < nnz:
        r = draw(integers(min_value=0, max_value=rows - 1))
        c = draw(integers(min_value=0, max_value=cols - 1))
        coords.add((r, c))

    row_ind, col_ind = zip(*coords)
    data = draw(
        arrays(dtype=dtype, shape=nnz, elements=floats(min_value=-1e3, max_value=1e3))
    )

    return coo_matrix((data, (row_ind, col_ind)), shape=(rows, cols))


# --- Property-Based Tests ---


@settings(deadline=1000, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(matrix_coo=sparse_matrices())
def test_property_matvec(matrix_coo: coo_matrix) -> None:
    """Property-based test for the matvec operation."""
    uri = "test_property_matvec_tiledb"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    try:
        convert_coo_to_tiledb(matrix_coo, uri, DataType.FLOAT64, overwrite=True)

        # Generate a compatible vector
        vector = np.random.rand(matrix_coo.shape[1])

        # Run the matvec operation with StreamMatrix
        ctx = tiledb.Ctx()
        stream_matrix = StreamMatrix(uri=uri, ctx=ctx)
        result_streammat = asyncio.run(stream_matrix.matvec(vector))

        # Calculate the expected result with SciPy
        result_scipy = matrix_coo.dot(vector)

        # Compare the results
        np.testing.assert_allclose(result_streammat, result_scipy, rtol=1e-5, atol=1e-8)

    finally:
        if os.path.exists(uri):
            shutil.rmtree(uri)


@settings(deadline=1000, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(matrix_coo=sparse_matrices())
def test_property_rmatvec(matrix_coo: coo_matrix) -> None:
    """Property-based test for the rmatvec operation."""
    uri = "test_property_rmatvec_tiledb"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    try:
        convert_coo_to_tiledb(matrix_coo, uri, DataType.FLOAT64, overwrite=True)

        # Generate a compatible vector
        vector = np.random.rand(matrix_coo.shape[0])

        # Run the rmatvec operation with StreamMatrix
        ctx = tiledb.Ctx()
        stream_matrix = StreamMatrix(uri=uri, ctx=ctx)
        result_streammat = asyncio.run(stream_matrix.rmatvec(vector))

        # Calculate the expected result with SciPy
        result_scipy = matrix_coo.T.dot(vector)

        # Compare the results
        np.testing.assert_allclose(result_streammat, result_scipy, rtol=1e-5, atol=1e-8)

    finally:
        if os.path.exists(uri):
            shutil.rmtree(uri)
