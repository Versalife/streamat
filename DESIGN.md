# StreamMat Architecture

This document outlines the architecture of the `streammat` project, a high-performance sparse matrix-vector operation library and server.

## 1. Overview

The `streammat` project is composed of two primary components:

1.  **A Command-Line Interface (CLI) for Data Conversion**: A tool named `streammat-convert` for converting sparse matrices from standard scientific formats into a specialized TileDB format.
2.  **A High-Performance API Server**: A FastAPI server that loads these TileDB matrices and exposes endpoints for high-performance matrix-vector multiplication.

The core design philosophy is to separate the slow, one-time data ingestion and conversion process from the fast, concurrent, online serving of mathematical operations.

## 2. Core Technologies

-   **Python 3.12+**: The primary programming language.
-   **TileDB**: Used as the storage backend for sparse matrices. TileDB is chosen for its ability to efficiently handle large, sparse, multi-dimensional arrays and its high-performance slicing capabilities.
-   **FastAPI**: The web framework for the API server. It's chosen for its high performance, asynchronous support, and automatic data validation and documentation features.
-   **Pydantic**: Used for data modeling, validation, and settings management. This ensures data integrity at the API boundary and for the metadata stored within TileDB arrays.
-   **Typer**: Used to create the command-line interface for the conversion tool.
-   **NumPy**: The fundamental package for numerical computation in Python.
-   **Asyncio**: Used extensively in the core logic and the server to achieve high concurrency for I/O-bound and parallelizable operations.
-   **aiorwlock**: Provides an asynchronous reader-writer lock to manage concurrent access to shared resources (the dictionary of loaded matrices) in the FastAPI server.

## 3. System Architecture

The system is divided into the offline conversion pipeline and the online API server.

### 3.1. Offline Conversion Pipeline (`streammat-convert`)

The conversion process is handled by the `streammat-convert` CLI tool, which is defined as a script entry point in `pyproject.toml`.

-   **File**: `conversion.py`
-   **Functionality**:
    -   Reads sparse matrices from various formats:
        -   Matrix Market (`.mtx`, `.mm`)
        -   Harwell-Boeing (`.hb`, `.rua`)
        -   Gene Cluster Text (`.gct`)
    -   It reads the input files in chunks to handle very large datasets that may not fit into memory.
    -   It creates a 2D sparse TileDB array with dimensions `dim_row` and `dim_col`.
    -   The non-zero values of the matrix are stored as a TileDB attribute named `attr_value`.
    -   Crucially, it stores metadata about the matrix (such as dimensions, number of non-zero elements, and data type) directly into the TileDB array's metadata. This metadata is defined by the `StreamMatConfig` Pydantic model.

This offline step ensures that the data is in an optimal format for the high-performance server, and all necessary metadata is co-located with the data.

### 3.2. Online API Server

The API server is a FastAPI application that provides RESTful endpoints for matrix operations.

-   **File**: `server.py`
-   **Core Components**:
    -   **FastAPI Application**: The main application instance. It uses a `lifespan` context manager to handle startup and shutdown events.
    -   **Global State**: A dictionary `server_state` holds the loaded `StreamMatrix` objects in memory, keyed by a user-defined `matrix_name`.
    -   **Reader-Writer Lock**: An `aiorwlock.RWLock` is used to protect access to the `server_state`.
        -   **Reader Lock**: Acquired for operations that do not modify the state, such as `matvec` and `rmatvec`. This allows multiple clients to perform calculations concurrently on the same or different matrices.
        -   **Writer Lock**: Acquired for operations that modify the state, such as loading or replacing a matrix. This ensures that no other operations can occur while the dictionary of matrices is being modified.
    -   **Exception Handling**: Custom exception handlers are in place to catch `StreamMatException` and other unexpected errors, returning structured JSON error responses to the client.

#### 3.2.1. Core Logic (`StreamMatrix` class)

-   **File**: `core.py`
-   **Functionality**:
    -   The `StreamMatrix` class is the core of the library. It encapsulates a TileDB array.
    -   Upon initialization, it opens the TileDB array and validates its schema and metadata against the `StreamMatConfig` model.
    -   **`matvec` (Matrix-Vector Multiplication)**:
        -   This method performs the operation `y = A * x`.
        -   It is implemented using `asyncio.gather` to execute the dot product for each row concurrently. The `_get_row_dot_product` helper function queries a single row from the TileDB array and computes its dot product with the input vector.
    -   **`rmatvec` (Adjoint Matrix-Vector Multiplication)**:
        -   This method performs the operation `y = A.T * x`.
        -   Similar to `matvec`, it uses `asyncio.gather` to execute the dot product for each column concurrently. The `_get_col_dot_product` helper function queries a single column from the TileDB array.

This asynchronous, parallel approach to the vector multiplications is the key to the system's performance.

### 3.3. Data Models

-   **File**: `models.py`
-   **Functionality**:
    -   Defines all Pydantic models used in the application, providing a single source of truth for data structures.
    -   `StreamMatConfig`: The metadata stored in the TileDB array.
    -   `VectorRequest`, `LoadMatrixRequest`: API request bodies.
    -   `ServerStatus`, `MatrixInfo`: API response bodies.
    -   `ErrorCode`, `ErrorDetail`, `ErrorResponse`: Structured error reporting.
    -   `StreamMatException`: A custom exception class used throughout the application for handling domain-specific errors.

## 4. API Endpoints

The server exposes the following RESTful endpoints:

-   `GET /api/v1/status`: Returns the server status and a list of all loaded matrices with their configurations.
-   `PUT /api/v1/matrix/{matrix_name}`: Loads a matrix from a given TileDB URI and assigns it a name. This is a "write" operation that requires an exclusive lock.
-   `POST /api/v1/matvec/{matrix_name}`: Performs matrix-vector multiplication. This is a "read" operation that can be executed concurrently.
-   `POST /api/v1/rmatvec/{matrix_name}`: Performs adjoint matrix-vector multiplication. This is also a "read" operation.

## 5. Example Usage

-   **File**: `example.py`
-   The `example.py` script provides a complete end-to-end demonstration of the system:
    1.  It programmatically creates a sample Matrix Market file.
    2.  It calls the `streammat-convert` CLI tool as a subprocess to convert this file into a TileDB array.
    3.  It initializes a `StreamMatrix` object from the newly created TileDB array.
    4.  It runs both `matvec` and `rmatvec` operations and verifies the results.
    5.  It cleans up the generated artifacts.

This script serves as a practical guide for users and a valuable integration test for developers.
