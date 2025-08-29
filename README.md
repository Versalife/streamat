# StreamMat: A High-Performance Sparse Matrix Operations Library

StreamMat is a Python library designed for high-performance, out-of-core sparse matrix operations. It uses TileDB as its storage backend to efficiently handle matrices that are too large to fit into memory. The library provides a FastAPI-based web server to expose matrix operations as a RESTful API.

## Key Features

- **Out-of-Core Computation**: Process massive sparse matrices that exceed available RAM.
- **High Performance**: Asynchronous, chunked operations to maximize throughput.
- **TileDB Backend**: Leverages the power of the TileDB array database for efficient sparse data storage and retrieval.
- **RESTful API**: Easy-to-use API for loading matrices, performing vector multiplications, and more.
- **Data Conversion**: A command-line tool to convert matrices from Matrix Market format to the StreamMat TileDB format.

## Installation

### From PyPI (Recommended)

Once published, you can install StreamMat directly from PyPI:

```bash
pip install streammat
```

### From Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/streamat.git
    cd streamat
    ```

2.  **Build and install the package:**
    ```bash
    uv build
    pip install dist/streammat-*.whl
    ```

## Usage

### 1. Converting a Matrix

StreamMat uses a specialized TileDB format. You can convert a standard Matrix Market (`.mtx`) file to this format using the `streammat-convert` CLI tool.

```bash
streammat-convert <input_mm_file.mtx> <output_tiledb_uri> [--dtype <data_type>] [--overwrite]
```

-   `<input_mm_file.mtx>`: Path to the input Matrix Market file.
-   `<output_tiledb_uri>`: The URI (local path) for the new TileDB array.
-   `--dtype`: The data type for the matrix values (e.g., `float32`, `float64`). Defaults to `float64`.
-   `--overwrite`: If specified, overwrite the output URI if it already exists.

**Example:**

```bash
streammat-convert example_matrix.mtx my_matrix_tiledb --dtype float32 --overwrite
```

### 2. Starting the Server

Once you have your matrix in the TileDB format, you can start the StreamMat server:

```bash
streammat-server
```

The server will be available at `http://127.0.0.1:8000`.

### 3. API Documentation

The server provides interactive API documentation (via Swagger UI) at `http://127.0.0.1:8000/docs`.

Here are some `curl` examples for interacting with the API:

#### Load a Matrix

This endpoint loads a TileDB matrix into the server's memory.

```bash
curl -X PUT "http://127.0.0.1:8000/api/v1/matrix/my_matrix" \
-H "Content-Type: application/json" \
-d 
{
  "uri": "my_matrix_tiledb"
}
```

#### Get Server Status

Check the server status and see which matrices are loaded.

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/status"
```

#### Perform Matrix-Vector Multiplication (matvec)

Calculates `y = A * x`.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/matvec/my_matrix" \
-H "Content-Type: application/json" \
-d 
{
  "vector": [1, 2, 3, 4]
}
```

#### Perform Adjoint Matrix-Vector Multiplication (rmatvec)

Calculates `y = A.T * x`.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/rmatvec/my_matrix" \
-H "Content-Type: application/json" \
-d 
{
  "vector": [10, 20, 30, 40]
}
```

#### Perform Matrix-Matrix Multiplication (matmul)

Calculates `C = A * B` and saves the result to a new TileDB array.

```bash
# First, load a second matrix
curl -X PUT "http://127.0.0.1:8000/api/v1/matrix/another_matrix" \
-H "Content-Type: application/json" \
-d 
{
  "uri": "my_matrix_tiledb"
}

# Then, perform the multiplication
curl -X POST "http://127.0.0.1:8000/api/v1/matmul/my_matrix/another_matrix" \
-H "Content-Type: application/json" \
-d 
{
  "output_uri": "result_matrix_tiledb"
}
```

#### Unload a Matrix

Remove a matrix from the server to free up resources.

```bash
curl -X DELETE "http://127.0.0.1:8000/api/v1/matrix/my_matrix"
```

## Complete Workflow Example

1.  **Clone the repository to run the example:**
    ```bash
    git clone https://github.com/your-username/streamat.git
    cd streamat
    ```

2.  **Install the development dependencies:**
    ```bash
    pip install .[dev]
    ```

3.  **Run the example script:**
    ```bash
    python streammat/example.py
    ```
