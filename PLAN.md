# StreamMat Development Plan

This document outlines a comprehensive plan to enhance the `streammat` project, evolving it into a more robust, feature-rich, and reliable library for out-of-core matrix operations.

## Phase 1: Stabilization and Testing

This phase focuses on fixing critical bugs, establishing a strong testing foundation, and ensuring the existing features are reliable.

### 1.1. Add Testing Framework and Dependencies
-   **Action**: Modify `pyproject.toml` to add a `[project.optional-dependencies]` group for `dev` dependencies.
-   **Dependencies**: `pytest`, `pytest-asyncio`, `requests`, `httpx`.
-   **Reasoning**: Establishes a standard, repeatable testing process. `pytest-asyncio` is required for testing async code, and `httpx` is the modern way to test a FastAPI app.

### 1.2. Fix Critical Bugs
-   **Bug 1: Hardcoded Vector `dtype` in Server**
    -   **File**: `server.py`
    -   **Problem**: The `matvec` and `rmatvec` endpoints hardcode the input vector's data type to `np.float64`, ignoring the matrix's actual `dtype`. This leads to incorrect calculations if the matrix is, for example, `float32`.
    -   **Fix**: The input vector must be cast to the `_numpy_dtype` of the `StreamMatrix` object.
-   **Bug 2: Silent Error Handling in Core**
    -   **File**: `core.py`
    -   **Problem**: The `_get_row_dot_product` and `_get_col_dot_product` methods catch all exceptions and return `0`, silently corrupting the final result vector if a single row/column fails.
    -   **Fix**: Remove the broad `try...except` block. Let exceptions propagate up to be caught by `asyncio.gather`. The main `matvec`/`rmatvec` methods should then handle these exceptions, ensuring that a failure in a sub-task fails the entire operation and returns an appropriate error to the user.

### 1.3. Create Test Suite
-   **Action**: Create a `tests/` directory.
-   **`tests/test_conversion.py`**:
    -   Test the `streammat-convert` CLI tool.
    -   Verify that it correctly converts a sample `.mtx` file to a TileDB array.
    -   Check if the metadata (`StreamMatConfig`) in the created array is correct.
-   **`tests/test_core.py`**:
    -   Unit test the `StreamMatrix` class.
    -   Test `matvec` and `rmatvec` against results from `numpy` and `scipy.sparse`.
    -   Test edge cases: empty matrix, all-zero matrix, dimension mismatch errors.
-   **`tests/test_server.py`**:
    -   Use FastAPI's `TestClient` (with `httpx`) to test the API.
    -   Test all endpoints: `GET /status`, `PUT /matrix/{name}`, `POST /matvec/{name}`, `POST /rmatvec/{name}`.
    -   Test successful cases and expected error responses (e.g., 404 for a missing matrix, 400 for dimension mismatch).

## Phase 2: Feature Enhancements

This phase focuses on improving usability, performance, and administration of the server.

### 2.1. Add Matrix Unloading
-   **Action**: Implement a `DELETE /api/v1/matrix/{matrix_name}` endpoint in `server.py`.
-   **Functionality**: This endpoint should safely remove a loaded matrix from the `server_state`, freeing up memory and resources. It requires a writer lock.
-   **Reasoning**: Essential for long-running servers to manage resources without a restart.

### 2.2. Improve Performance with Chunking
-   **Action**: Refactor `matvec` and `rmatvec` in `core.py`.
-   **Problem**: Creating an `asyncio` task for every single row/column can be inefficient due to high overhead for very large matrices.
-   **Fix**: Introduce a `chunk_size` parameter (e.g., 1000). The methods will process the matrix in chunks, creating a more manageable number of concurrent tasks. This involves querying data for multiple rows/columns at once from TileDB, which is more efficient.

### 2.3. Add Server Configuration via Environment Variables
-   **Action**: Use Pydantic's `BaseSettings` to create a configuration model.
-   **Functionality**: Manage server settings like `HOST` and `PORT` through environment variables.
-   **File**: Create a new `config.py` and use it in `server.py`.
-   **Reasoning**: Standard practice for deploying applications, making the server more flexible.

## Phase 3: Major New Features

This phase introduces significant new capabilities to the library.

### 3.1. Implement Matrix-Matrix Multiplication
-   **Action**: Add a `matmul(A, B)` feature.
-   **API**: Create a new endpoint `POST /api/v1/matmul/{matrix_a_name}/{matrix_b_name}`. The request body could specify the URI for the output matrix `C`.
-   **Core Logic**:
    -   The core `matmul` function will take two `StreamMatrix` objects, `A` and `B`.
    -   It will create a new, writable TileDB array `C` for the result.
    -   The calculation `C = A @ B` will be performed by iterating through the columns of `B`, performing a `matvec` (`A @ b_col`), and writing the resulting column vector to `C`.
    -   This process should be chunked and parallelized for performance.
-   **Reasoning**: This is the most requested feature for a "matmul" library and a natural extension of `matvec`.

### 3.2. Populate README.md
-   **Action**: Write a comprehensive `README.md`.
-   **Content**:
    -   Project description.
    -   Installation instructions (`pip install .` and `pip install .[dev]`).
    -   Usage guide for `streammat-convert`.
    -   API documentation with `curl` examples for each endpoint.
    -   A complete example of the workflow.
-   **Reasoning**: Makes the project accessible and usable for others.

## Phase 4: Future Work (Out of Scope for Immediate Implementation)

-   **More Input Formats**: Support for direct conversion from `scipy.sparse` objects or `numpy` arrays.
-   **Additional Linear Algebra Operations**: Element-wise operations, scalar multiplication, matrix transpose, etc.
-   **API Authentication**: Add a simple API key mechanism for security.
-   **Improved Logging**: Allow configurable log levels and formats.

## Phase 5: User-Requested Enhancements (Sept 2025)

This phase addresses a set of user-requested features to improve usability, resource management, and API ergonomics.

### 5.1. Unload with Deletion
- **Action**: Modify the `DELETE /api/v1/matrix/{matrix_name}` endpoint in `server.py`.
- **Functionality**: When a matrix is unloaded, its underlying TileDB array will be deleted from disk.
- **Reasoning**: To ensure that "unloading" a matrix actually frees up disk space, making resource management more intuitive.

### 5.2. Progress Communication for Matrix Loading
- **Action**: Explore and implement a progress communication mechanism for the `PUT /api/v1/matrix/{matrix_name}` endpoint.
- **Functionality**: The server will send status updates to the client during long-running matrix load/conversion operations.
- **Implementation**:
    - **Option 1**: Use `fastapi.responses.StreamingResponse`.
    - **Option 2**: Implement a WebSocket-based notification system if streaming is insufficient.
- **Reasoning**: To improve the user experience by providing feedback on slow operations, preventing timeouts and uncertainty.

### 5.3. API for Shape Discovery
- **Action**: Enhance the `MatrixInfo` model and the `/api/v1/status` endpoint.
- **Functionality**: The API will expose the expected input and output shapes for vector operations (`matvec`, `rmatvec`).
- **Reasoning**: To make the API more discoverable and prevent users from having to trigger a dimension mismatch error to find out the expected vector size.

### 5.4. Ergonomic API for Sparse Data
- **Action**: Redesign API models (`VectorRequest`) and core logic to handle sparse data formats.
- **Functionality**: Allow users to pass sparse vectors (e.g., as a dictionary of indices and values) instead of dense arrays.
- **Reasoning**: To make the API more ergonomic and efficient, especially for users working with very large, sparse vectors where listing all the zeros is impractical.