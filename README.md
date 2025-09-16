# StreamMat: A High-Performance Sparse Matrix Operations Library

[![CI](https://github.com/Versalife/streamat/actions/workflows/ci.yml/badge.svg)](https://github.com/Versalife/streamat/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

StreamMat is a Python library designed for high-performance, out-of-core sparse matrix operations. It uses TileDB as its storage backend to efficiently handle matrices that are too large to fit into memory. The library provides a FastAPI-based web server to expose matrix operations as a RESTful API.

## Key Features

- **Out-of-Core Computation**: Process massive sparse matrices that exceed available RAM.
- **High Performance**: Asynchronous, chunked operations to maximize throughput.
- **TileDB Backend**: Leverages the power of the TileDB array database for efficient sparse data storage and retrieval.
- **RESTful API**: Easy-to-use API for loading matrices, performing vector multiplications, and more.
- **On-the-Fly Data Conversion**: Automatically converts matrices from popular formats (like Matrix Market) and downloads from URLs.

## Installation

### From PyPI (Recommended)

Once published, you can install StreamMat directly from PyPI:

```bash
pip install streammat
```

### From Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Versalife/streamat.git
    cd streamat
    ```

2.  **Build and install the package:**
    ```bash
    uv build
    pip install dist/streammat-*.whl
    ```

## Usage

### 1. Converting a Matrix (Optional)

StreamMat can automatically convert matrices on the fly. However, you can still pre-convert a standard Matrix Market (`.mtx`) file to the TileDB format using the `streammat-convert` CLI tool.

```bash
streammat-convert <input_mm_file.mtx> <output_tiledb_uri> [--dtype <data_type>] [--overwrite]
```

### 2. Starting the Server

```bash
streammat-server
```

The server will be available at `http://127.0.0.1:8000`.

### 3. API Documentation

The server provides interactive API documentation (via Swagger UI) at `http://127.0.0.1:8000/docs`.

Here are some `curl` examples for interacting with the API:

#### Load a Matrix

This endpoint loads a matrix into the server's memory. It can be a path to a local TileDB array, a local file that needs conversion (e.g., `.mtx`), or a URL to a file.

The endpoint returns a stream of server-sent events to track progress.

```bash
# Load from a local TileDB array (or a file that needs conversion)
curl -N -X PUT "http://127.0.0.1:8000/api/v1/matrix/my_matrix" \
-H "Content-Type: application/json" \
-d '{
  "uri": "my_matrix_tiledb"
}'

# Example Output Stream:
# data: {"status": "converting", "progress": 50.00}
# data: {"status": "converting", "progress": 100.00}
# data: {"status": "loading_complete", "uri": "/path/to/cache/....tiledb"}
```

#### Get Server Status & Shape Discovery

Check the server status, see which matrices are loaded, and discover the expected shapes for operations.

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/status"
```

The response will include an `operations` field for each matrix, detailing the `input_shape` and `output_shape` for `matvec` and `rmatvec`.

#### Perform Matrix-Vector Multiplication (matvec)

Calculates `y = A * x`. You can provide the vector as a dense list or a sparse object.

```bash
# Dense vector
curl -X POST "http://127.0.0.1:8000/api/v1/matvec/my_matrix" \
-H "Content-Type: application/json" \
-d '{
  "vector": [1, 2, 3, 4]
}'

# Sparse vector (equivalent to [1, 0, 0, 4])
curl -X POST "http://127.0.0.1:8000/api/v1/matvec/my_matrix" \
-H "Content-Type: application/json" \
-d '{
  "vector": {
    "indices": [0, 3],
    "values": [1.0, 4.0]
  }
}'
```

#### Unload a Matrix

Remove a matrix from the server and **delete its data from the disk cache**.

```bash
curl -X DELETE "http://127.0.0.1:8000/api/v1/matrix/my_matrix"
```

## Complete Workflow Example

1.  **Clone the repository to run the example:**
    ```bash
    git clone https://github.com/Versalife/streamat.git
    cd streamat
    ```

2.  **Install the development dependencies:**
    ```bash
    uv pip install --system '.[dev]'
    ```

3.  **Run the example script:**
    ```bash
    python streamat/example.py
    ```