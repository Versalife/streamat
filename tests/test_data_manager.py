import asyncio
import http.server
import socketserver
import threading
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from streammat.server import app

MM_CONTENT = """%%MatrixMarket matrix coordinate real general
% A simple 2x2 sparse matrix
2 2 2
1 1 1.0
2 2 2.0
"""


@pytest.fixture(scope="module")
def http_server():
    """Fixture to run a simple HTTP server in a background thread."""
    PORT = 8080
    handler = http.server.SimpleHTTPRequestHandler
    httpd = None

    # Create a temporary file to serve
    temp_dir = Path("/tmp/streammat_test_server")
    temp_dir.mkdir(exist_ok=True)
    (temp_dir / "test.mtx").write_text(MM_CONTENT)

    class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(temp_dir), **kwargs)

    # Try to start the server, handling "address already in use"
    for i in range(5):
        try:
            httpd = socketserver.TCPServer(("", PORT + i), MyHttpRequestHandler)
            break
        except OSError:
            continue

    if httpd is None:
        pytest.fail("Could not find an open port for the test HTTP server.")

    final_port = httpd.server_address[1]
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    yield f"http://127.0.0.1:{final_port}/test.mtx"

    # Teardown
    httpd.shutdown()
    httpd.server_close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_load_matrix_from_url(http_server):
    """Tests loading a matrix from a URL with on-the-fly conversion."""
    matrix_name = "remote_matrix"
    request_body = {"uri": http_server}

    with TestClient(app) as client:
        # 1. Load the matrix from the URL, consuming the stream
        with client.stream("PUT", f"/api/v1/matrix/{matrix_name}", json=request_body) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if "loading_complete" in line:
                    break

        # 2. Verify the matrix is loaded
        status_response = client.get("/api/v1/status")
        assert status_response.status_code == 200
        loaded_matrices = status_response.json()["loaded_matrices"]
        assert matrix_name in loaded_matrices
        assert loaded_matrices[matrix_name]["config"]["rows"] == 2
        assert loaded_matrices[matrix_name]["config"]["cols"] == 2

        # 3. Perform a matvec operation to confirm it works
        matvec_request = {"vector": [1.0, 2.0]}
        matvec_response = client.post(f"/api/v1/matvec/{matrix_name}", json=matvec_request)
        assert matvec_response.status_code == 200
        np.testing.assert_allclose(matvec_response.json()["result"], [1.0, 4.0])

        # 4. Check that the temporary downloaded file has been cleaned up
        temp_dir = Path(tempfile.gettempdir())
        downloaded_files = list(temp_dir.glob("test.mtx*"))
        assert not downloaded_files, f"Temporary file was not cleaned up: {downloaded_files}"
