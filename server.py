"""
The FastAPI web server providing a RESTful API for StreamMat operations.
"""
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
from aiorwlock import RWLock
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger

from core import StreamMatrix
from logging_config import setup_logging
from models import (ErrorCode, ErrorDetail, ErrorResponse, LoadMatrixRequest,
                    MatrixInfo, ServerStatus, StreamMatException,
                    VectorRequest)

# Global state for the server
server_state: Dict[str, any] = {
    "loaded_matrices": {},
    "lock": RWLock(),  # Reader-Writer lock for concurrent access to the matrices dict
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on server startup
    setup_logging()
    logger.info("StreamMat server is starting up...")
    yield
    # Code to run on server shutdown
    logger.info("StreamMat server is shutting down...")
    server_state["loaded_matrices"].clear()


app = FastAPI(
    title="StreamMat Server",
    description="A high-performance API for sparse matrix operations using TileDB.",
    version="1.1.0",
    lifespan=lifespan
)


# --- Exception Handling ---
@app.exception_handler(StreamMatException)
async def streammat_exception_handler(request: Request, exc: StreamMatException):
    """Handles custom StreamMatExceptions and returns a structured JSON error response."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if exc.code is ErrorCode.MATRIX_NOT_FOUND:
        status_code = status.HTTP_404_NOT_FOUND
    elif exc.code in [ErrorCode.DIMENSION_MISMATCH, ErrorCode.INVALID_REQUEST]:
        status_code = status.HTTP_400_BAD_REQUEST
    elif exc.code in [ErrorCode.TILEDB_ERROR, ErrorCode.INVALID_TILEDB_SCHEMA]:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    logger.error(f"Error processing request {request.url.path}: {exc}")
    error_detail = ErrorDetail(code=exc.code, message=exc.message)
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error_detail).model_dump()
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Handles any other unhandled exceptions."""
    logger.exception(f"Unhandled exception for request {request.url.path}")
    error_detail = ErrorDetail(
        code=ErrorCode.INTERNAL_SERVER_ERROR,
        message="An unexpected internal server error occurred."
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error=error_detail).model_dump()
    )


# --- API Endpoints ---
@app.get(
    "/api/v1/status",
    response_model=ServerStatus,
    summary="Get Server Status and Loaded Matrices"
)
async def get_status():
    """Returns the current status of the server and details of all loaded matrices."""
    logger.info("Status endpoint requested.")
    async with server_state["lock"].reader_lock:
        matrices_info = {
            name: MatrixInfo(uri=matrix.uri, config=matrix.config)
            for name, matrix in server_state["loaded_matrices"].items()
        }
    return ServerStatus(
        server_status="running",
        loaded_matrices=matrices_info
    )


@app.put(
    "/api/v1/matrix/{matrix_name}",
    status_code=status.HTTP_201_CREATED,
    summary="Load or Replace a Matrix"
)
async def load_matrix(matrix_name: str, request: LoadMatrixRequest):
    """
    Loads a new matrix from a TileDB array URI or replaces an existing one.
    This is a write operation and requires an exclusive lock.
    """
    logger.info(f"Request to load matrix '{matrix_name}' from URI '{request.uri}'.")
    await _load_matrix_into_state(matrix_name, request.uri)
    logger.success(f"Successfully loaded matrix '{matrix_name}'.")
    return {"message": f"Matrix '{matrix_name}' loaded successfully from '{request.uri}'."}


async def _load_matrix_into_state(matrix_name: str, uri: str):
    """Helper function to load a matrix and place it into the server state."""
    import tiledb

    try:
        matrix = StreamMatrix(uri=uri, ctx=tiledb.Ctx())
        async with server_state["lock"].writer_lock:
            server_state["loaded_matrices"][matrix_name] = matrix
    except StreamMatException as e:
        logger.error(f"Failed to load matrix '{matrix_name}': {e.message}")
        raise  # Re-raise to be handled by the exception handler
    except Exception as e:
        logger.exception("An unexpected error occurred during matrix loading.")
        raise StreamMatException(ErrorCode.INTERNAL_SERVER_ERROR, str(e)) from e


@app.post(
    "/api/v1/matvec/{matrix_name}",
    summary="Matrix-Vector Multiplication"
)
async def perform_matvec(matrix_name: str, request: VectorRequest):
    """Performs matrix-vector multiplication (y = A * x)."""
    logger.info(f"Received matvec request for matrix '{matrix_name}'.")
    async with server_state["lock"].reader_lock:
        if matrix_name not in server_state["loaded_matrices"]:
            raise StreamMatException(ErrorCode.MATRIX_NOT_FOUND, f"Matrix '{matrix_name}' not found.")
        matrix = server_state["loaded_matrices"][matrix_name]

    input_vector = np.array(request.vector, dtype=np.float64)
    result_vector = await matrix.matvec(input_vector)

    return {"result": result_vector.tolist()}


@app.post(
    "/api/v1/rmatvec/{matrix_name}",
    summary="Adjoint Matrix-Vector Multiplication"
)
async def perform_rmatvec(matrix_name: str, request: VectorRequest):
    """Performs adjoint matrix-vector multiplication (y = A.T * x)."""
    logger.info(f"Received rmatvec request for matrix '{matrix_name}'.")
    async with server_state["lock"].reader_lock:
        if matrix_name not in server_state["loaded_matrices"]:
            raise StreamMatException(ErrorCode.MATRIX_NOT_FOUND, f"Matrix '{matrix_name}' not found.")
        matrix = server_state["loaded_matrices"][matrix_name]

    input_vector = np.array(request.vector, dtype=np.float64)
    result_vector = await matrix.rmatvec(input_vector)

    return {"result": result_vector.tolist()}
