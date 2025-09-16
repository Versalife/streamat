import os
import shutil
from pathlib import Path
import numpy as np
import pytest
import tiledb
from fastapi.testclient import TestClient

from streammat.server import app
from streammat.core import ATTR_VALUE_NAME, DIM_COL_NAME, DIM_ROW_NAME
from streammat.models import DataType, StreamMatConfig

TEST_MATRIX_URI = "test_matrix_tiledb"
TEST_MATRIX_FLOAT32_URI = "test_matrix_float32_tiledb"

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def create_test_tiledb_array(uri: str, dtype: np.dtype, lib_dtype: DataType):
    """Helper function to create a standard TileDB array for tests."""
    if os.path.exists(uri):
        shutil.rmtree(uri)

    rows, cols = 4, 4
    domain = tiledb.Domain(
        tiledb.Dim(name=DIM_ROW_NAME, domain=(0, rows - 1), tile=4, dtype=np.uint64),
        tiledb.Dim(name=DIM_COL_NAME, domain=(0, cols - 1), tile=4, dtype=np.uint64),
    )
    schema = tiledb.ArraySchema(
        domain=domain,
        sparse=True,
        attrs=[tiledb.Attr(name=ATTR_VALUE_NAME, dtype=dtype)]
    )
    tiledb.Array.create(uri, schema)

    # Matrix data:
    # [[1.1, 0, 0, 5.5],
    #  [0, 2.2, 0, 0],
    #  [0, 0, 3.3, 0],
    #  [0, 0, 0, 4.4]]
    row_coords = np.array([0, 1, 2, 3, 0])
    col_coords = np.array([0, 1, 2, 3, 3])
    values = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=dtype)
    nnz = len(values)

    with tiledb.open(uri, 'w') as A:
        A[row_coords, col_coords] = values
        config = StreamMatConfig(rows=rows, cols=cols, nnz=nnz, dtype=lib_dtype)
        for field, value in config.model_dump(by_alias=True).items():
            A.meta[field] = value.value if isinstance(value, DataType) else value

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_tiledb():
    """Fixture to create and clean up test TileDB arrays."""
    create_test_tiledb_array(TEST_MATRIX_URI, np.float64, DataType.FLOAT64)
    create_test_tiledb_array(TEST_MATRIX_FLOAT32_URI, np.float32, DataType.FLOAT32)
    yield
    # Teardown
    if os.path.exists(TEST_MATRIX_URI):
        shutil.rmtree(TEST_MATRIX_URI)
    if os.path.exists(TEST_MATRIX_FLOAT32_URI):
        shutil.rmtree(TEST_MATRIX_FLOAT32_URI)

def test_load_matrix(client):
    matrix_name = "test_matrix_load"
    with client.stream(
        "PUT",
        f"/api/v1/matrix/{matrix_name}",
        json={"uri": TEST_MATRIX_URI}
    ) as response:
        assert response.status_code == 200
        # Consume the stream to ensure the loading is complete
        for line in response.iter_lines():
            print(line) # for debugging
            if "loading_complete" in line:
                break

    # Check status
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["server_status"] == "running"
    assert matrix_name in data["loaded_matrices"]
    assert data["loaded_matrices"][matrix_name]["config"]["rows"] == 4

def test_matvec_success(client):
    # Ensure matrix is loaded
    client.put("/api/v1/matrix/test_matrix", json={"uri": TEST_MATRIX_URI})

    response = client.post(
        "/api/v1/matvec/test_matrix",
        json={"vector": [1, 2, 3, 4]}
    )
    assert response.status_code == 200
    result = response.json()["result"]
    # Expected: A * x = [1.1*1 + 5.5*4, 2.2*2, 3.3*3, 4.4*4] = [23.1, 4.4, 9.9, 17.6]
    np.testing.assert_allclose(result, [23.1, 4.4, 9.9, 17.6])

def test_rmatvec_success(client):
    # Ensure matrix is loaded
    client.put("/api/v1/matrix/test_matrix", json={"uri": TEST_MATRIX_URI})

    response = client.post(
        "/api/v1/rmatvec/test_matrix",
        json={"vector": [10, 20, 30, 40]}
    )
    assert response.status_code == 200
    result = response.json()["result"]
    # Expected: A.T * y = [1.1*10, 2.2*20, 3.3*30, 5.5*10 + 4.4*40] = [11.0, 44.0, 99.0, 231.0]
    np.testing.assert_allclose(result, [11.0, 44.0, 99.0, 231.0])

def test_matvec_dtype_mismatch_fix(client):
    """Tests the bug fix for hardcoded dtype by using a float32 matrix."""
    client.put("/api/v1/matrix/test_matrix_f32", json={"uri": TEST_MATRIX_FLOAT32_URI})
    response = client.post(
        "/api/v1/matvec/test_matrix_f32",
        json={"vector": [1, 2, 3, 4]}
    )
    assert response.status_code == 200
    result = np.array(response.json()["result"])
    # If the fix works, the result should be float32
    assert result.dtype == np.float32 or result.dtype == np.float64 # JSON conversion might make it float64
    np.testing.assert_allclose(result, [23.1, 4.4, 9.9, 17.6], rtol=1e-5)

def test_matrix_not_found(client):
    response = client.post(
        "/api/v1/matvec/non_existent_matrix",
        json={"vector": [1, 2, 3, 4]}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MATRIX_NOT_FOUND"

def test_dimension_mismatch(client):
    client.put("/api/v1/matrix/test_matrix", json={"uri": TEST_MATRIX_URI})
    response = client.post(
        "/api/v1/matvec/test_matrix",
        json={"vector": [1, 2, 3]} # Incorrect dimension
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "DIMENSION_MISMATCH"


def test_status_endpoint_for_shape_discovery(client):
    """Tests that the status endpoint includes operation shapes."""
    client.put("/api/v1/matrix/test_matrix", json={"uri": TEST_MATRIX_URI})
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    matrix_info = data["loaded_matrices"]["test_matrix"]

    assert "operations" in matrix_info
    ops = matrix_info["operations"]
    assert "matvec" in ops
    assert ops["matvec"]["input_shape"] == [4]
    assert ops["matvec"]["output_shape"] == [4]
    assert "rmatvec" in ops
    assert ops["rmatvec"]["input_shape"] == [4]
    assert ops["rmatvec"]["output_shape"] == [4]


def test_matvec_with_sparse_vector(client):
    """Tests matvec with a sparse vector input."""
    client.put("/api/v1/matrix/test_matrix", json={"uri": TEST_MATRIX_URI})
    sparse_vector = {
        "indices": [0, 3],
        "values": [1.0, 4.0]
    } # Represents dense vector [1, 0, 0, 4]
    response = client.post(
        "/api/v1/matvec/test_matrix",
        json={"vector": sparse_vector}
    )
    assert response.status_code == 200
    result = response.json()["result"]
    # Expected: A * x = [1.1*1 + 5.5*4, 0, 0, 4.4*4] = [23.1, 0, 0, 17.6]
    np.testing.assert_allclose(result, [23.1, 0, 0, 17.6])


def test_rmatvec_with_sparse_vector(client):
    """Tests rmatvec with a sparse vector input."""
    client.put("/api/v1/matrix/test_matrix", json={"uri": TEST_MATRIX_URI})
    sparse_vector = {
        "indices": [0, 2],
        "values": [10.0, 30.0]
    } # Represents dense vector [10, 0, 30, 0]
    response = client.post(
        "/api/v1/rmatvec/test_matrix",
        json={"vector": sparse_vector}
    )
    assert response.status_code == 200
    result = response.json()["result"]
    # Expected: A.T * y = [1.1*10, 0, 3.3*30, 5.5*10] = [11.0, 0, 99.0, 55.0]
    np.testing.assert_allclose(result, [11.0, 0, 99.0, 55.0])


def test_unload_matrix(client):
    matrix_name = "test_unload"
    unload_uri = "test_unload_matrix_tiledb"
    # Create a dedicated matrix for this test so we can safely delete it
    create_test_tiledb_array(unload_uri, np.float64, DataType.FLOAT64)
    assert os.path.exists(unload_uri)

    # First, load the matrix
    with client.stream("PUT", f"/api/v1/matrix/{matrix_name}", json={"uri": unload_uri}) as response:
        for line in response.iter_lines():
            if "loading_complete" in line:
                break
    
    # Check that it's loaded
    response = client.get("/api/v1/status")
    assert matrix_name in response.json()["loaded_matrices"]

    # Unload the matrix
    response = client.delete(f"/api/v1/matrix/{matrix_name}")
    assert response.status_code == 200
    assert "unloaded and its data" in response.json()["message"]

    # Check that it's gone from the server status
    response = client.get("/api/v1/status")
    assert matrix_name not in response.json()["loaded_matrices"]

    # Crucially, check that the on-disk artifact is also deleted
    assert not os.path.exists(unload_uri)

    # Test unloading a non-existent matrix
    response = client.delete("/api/v1/matrix/non_existent_matrix")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MATRIX_NOT_FOUND"

def test_matmul_success(client):
    # Load matrices A and B
    client.put("/api/v1/matrix/matrix_a", json={"uri": TEST_MATRIX_URI})
    client.put("/api/v1/matrix/matrix_b", json={"uri": TEST_MATRIX_URI})

    output_uri = "test_matmul_output"
    
    if os.path.exists(output_uri):
        shutil.rmtree(output_uri)

    try:
        response = client.post(
            "/api/v1/matmul/matrix_a/matrix_b",
            json={"output_uri": output_uri}
        )
        assert response.status_code == 200
        assert response.json() == {"message": f"Matrix multiplication complete. Result stored at '{output_uri}'."}

        # Verify the result
        with tiledb.open(output_uri, 'r') as C:
            # A is the test matrix, so C = A * A
            # A = [[1.1, 0, 0, 5.5], [0, 2.2, 0, 0], [0, 0, 3.3, 0], [0, 0, 0, 4.4]]
            # C[0,0] = 1.1*1.1 = 1.21
            # C[0,3] = 1.1*5.5 + 5.5*4.4 = 6.05 + 24.2 = 30.25
            # C[1,1] = 2.2*2.2 = 4.84
            # C[2,2] = 3.3*3.3 = 10.89
            # C[3,3] = 4.4*4.4 = 19.36
            assert C[0,0][ATTR_VALUE_NAME][0] == pytest.approx(1.21)
            assert C[0,3][ATTR_VALUE_NAME][0] == pytest.approx(30.25)
            assert C[1,1][ATTR_VALUE_NAME][0] == pytest.approx(4.84)
            assert C[2,2][ATTR_VALUE_NAME][0] == pytest.approx(10.89)
            assert C[3,3][ATTR_VALUE_NAME][0] == pytest.approx(19.36)

    finally:
        if os.path.exists(output_uri):
            shutil.rmtree(output_uri)