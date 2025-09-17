"The FastAPI web server providing a RESTful API for StreamMat operations."

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TypedDict

import numpy as np
from aiorwlock import RWLock
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from .config import settings
from .core import StreamMatrix
from .data_manager import DataManager
from .logging_config import setup_logging
from .models import (
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    LoadMatrixRequest,
    MatMulRequest,
    MatrixInfo,
    OperationInfo,
    ServerStatus,
    SparseVector,
    StreamMatException,
    VectorRequest,
    VectorResponse,
)


# Global state for the server
class ServerState(TypedDict):
    loaded_matrices: dict[str, StreamMatrix]
    lock: RWLock
    data_manager: DataManager


server_state: ServerState


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Code to run on server startup
    global server_state
    setup_logging()
    logger.info("StreamMat server is starting up...")
    server_state = {
        "loaded_matrices": {},
        "lock": RWLock(),
        "data_manager": DataManager(),
    }
    yield
    # Code to run on server shutdown
    logger.info("StreamMat server is shutting down...")
    server_state["loaded_matrices"].clear()


app = FastAPI(
    title="StreamMat Server",
    description="A high-performance API for sparse matrix operations using TileDB.",
    version="1.1.0",
    lifespan=lifespan,
)


# --- Exception Handling ---
@app.exception_handler(StreamMatException)
async def streammat_exception_handler(
    request: Request, exc: StreamMatException
) -> JSONResponse:
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
        status_code=status_code, content=ErrorResponse(error=error_detail).model_dump()
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handles any other unhandled exceptions."""
    logger.exception(f"Unhandled exception for request {request.url.path}")
    error_detail = ErrorDetail(
        code=ErrorCode.INTERNAL_SERVER_ERROR,
        message="An unexpected internal server error occurred.",
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error=error_detail).model_dump(),
    )


# --- API Endpoints ---
@app.get(
    "/api/v1/status",
    response_model=ServerStatus,
    summary="Get Server Status and Loaded Matrices",
)
async def get_status() -> ServerStatus:
    """Returns the current status of the server and details of all loaded matrices."""
    logger.info("Status endpoint requested.")
    async with server_state["lock"].reader_lock:
        matrices_info = {}
        for name, matrix in server_state["loaded_matrices"].items():
            matrices_info[name] = MatrixInfo(
                uri=matrix.uri,
                config=matrix.config,
                operations={
                    "matvec": OperationInfo(
                        input_shape=[matrix.config.cols],
                        output_shape=[matrix.config.rows],
                    ),
                    "rmatvec": OperationInfo(
                        input_shape=[matrix.config.rows],
                        output_shape=[matrix.config.cols],
                    ),
                },
            )
    return ServerStatus(server_status="running", loaded_matrices=matrices_info)


@app.put(
    "/api/v1/matrix/{matrix_name}",
    summary="Load or Replace a Matrix with Progress Stream",
)
async def load_matrix(
    matrix_name: str, request: LoadMatrixRequest
) -> StreamingResponse:
    """
    Loads a new matrix, streaming progress updates.
    If the URI is not a TileDB array, it will be downloaded and/or converted
    automatically.
    """
    logger.info(f"Request to load matrix '{matrix_name}' from URI '{request.uri}'.")
    data_manager = server_state["data_manager"]

    async def stream_progress() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[str] = asyncio.Queue()

        async def progress_callback(message: str) -> None:
            logger.info(f"Streaming progress for {matrix_name}: {message}")
            await queue.put(message)

        async def provision_and_load() -> None:
            final_uri = None
            try:
                final_uri = await data_manager.provision_matrix(
                    request.uri, matrix_name, progress_callback
                )
                await _load_matrix_into_state(matrix_name, final_uri)
                await progress_callback(
                    f'{{"status": "loading_complete", "uri": "{final_uri}"}}'
                )
            except Exception as e:
                error_msg = f"Failed to provision or load matrix: {e}"
                logger.error(error_msg)
                await progress_callback(
                    f'{{"status": "error", "message": "{error_msg}"}}'
                )
            finally:
                await progress_callback("END_OF_STREAM")

        # Start the provisioning and loading process in the background
        asyncio.create_task(provision_and_load())

        # Yield messages from the queue until the end signal is received
        while True:
            message = await queue.get()
            if message == "END_OF_STREAM":
                break
            yield f"data: {message}\n\n"

    return StreamingResponse(stream_progress(), media_type="text/event-stream")


@app.delete(
    "/api/v1/matrix/{matrix_name}",
    status_code=status.HTTP_200_OK,
    summary="Unload a Matrix",
    response_model=None,
)
async def unload_matrix(matrix_name: str) -> dict[str, str] | JSONResponse:
    """
    Unloads a matrix from the server and deletes its TileDB array from disk.
    """
    import tiledb

    logger.info(f"Request to unload and delete matrix '{matrix_name}'.")

    async with server_state["lock"].writer_lock:
        if matrix_name not in server_state["loaded_matrices"]:
            raise StreamMatException(
                ErrorCode.MATRIX_NOT_FOUND, f"Matrix '{matrix_name}' not found."
            )

        matrix_to_delete = server_state["loaded_matrices"][matrix_name]
        uri_to_delete = matrix_to_delete.uri

        del server_state["loaded_matrices"][matrix_name]
        logger.success(
            f"Successfully unloaded matrix '{matrix_name}' from server memory."
        )

    try:
        tiledb.remove(uri_to_delete)
        logger.success(f"Successfully deleted TileDB array at '{uri_to_delete}'.")
        return {
            "message": (
                f"Matrix '{matrix_name}' unloaded and its data at '{uri_to_delete}' deleted."
            )
        }
    except Exception as e:
        logger.error(f"Failed to delete TileDB array at '{uri_to_delete}': {e}")
        # We still return a success because the primary goal (unloading from memory) was
        # achieved.
        # The error is logged for the operator to handle.
        return JSONResponse(
            status_code=status.HTTP_207_MULTI_STATUS,
            content={
                "message": (
                    f"Matrix '{matrix_name}' unloaded from memory, but failed to "
                    f"delete the on-disk array at '{uri_to_delete}'. Please check server logs."
                ),
            },
        )


async def _load_matrix_into_state(matrix_name: str, uri: str) -> None:
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


def _prepare_input_vector(
    vector_request: VectorRequest, expected_dim: int, dtype: np.dtype
) -> np.ndarray:
    """Prepares a numpy array from a dense or sparse vector request."""
    if isinstance(vector_request.vector, SparseVector):
        if not vector_request.vector.indices or not vector_request.vector.values:
            raise StreamMatException(
                ErrorCode.INVALID_REQUEST,
                "Sparse vector must have both indices and values.",
            )
        if len(vector_request.vector.indices) != len(vector_request.vector.values):
            raise StreamMatException(
                ErrorCode.INVALID_REQUEST,
                "Sparse vector indices and values must have the same length.",
            )

        input_vector = np.zeros(expected_dim, dtype=dtype)
        # Ensure indices are within bounds
        max_index = max(vector_request.vector.indices)
        if max_index >= expected_dim:
            raise StreamMatException(
                ErrorCode.DIMENSION_MISMATCH,
                f"Index {max_index} in sparse vector is out of bounds for dimension "
                f"{expected_dim}.",
            )

        input_vector[vector_request.vector.indices] = vector_request.vector.values
        return input_vector
    else:
        return np.array(vector_request.vector, dtype=dtype)


@app.post(
    "/api/v1/matvec/{matrix_name}",
    response_model=VectorResponse,
    summary="Matrix-Vector Multiplication",
)
async def perform_matvec(matrix_name: str, request: VectorRequest) -> VectorResponse:
    """Performs matrix-vector multiplication (y = A * x)."""
    logger.info(f"Received matvec request for matrix '{matrix_name}'.")
    async with server_state["lock"].reader_lock:
        if matrix_name not in server_state["loaded_matrices"]:
            raise StreamMatException(
                ErrorCode.MATRIX_NOT_FOUND, f"Matrix '{matrix_name}' not found."
            )
        matrix = server_state["loaded_matrices"][matrix_name]

    input_vector = _prepare_input_vector(
        request, matrix.config.cols, np.dtype(matrix._numpy_dtype)
    )
    result_vector = await matrix.matvec(input_vector)

    return VectorResponse(result=result_vector.tolist())


@app.post(
    "/api/v1/rmatvec/{matrix_name}",
    response_model=VectorResponse,
    summary="Adjoint Matrix-Vector Multiplication",
)
async def perform_rmatvec(matrix_name: str, request: VectorRequest) -> VectorResponse:
    """Performs adjoint matrix-vector multiplication (y = A.T * x)."""
    logger.info(f"Received rmatvec request for matrix '{matrix_name}'.")
    async with server_state["lock"].reader_lock:
        if matrix_name not in server_state["loaded_matrices"]:
            raise StreamMatException(
                ErrorCode.MATRIX_NOT_FOUND, f"Matrix '{matrix_name}' not found."
            )
        matrix = server_state["loaded_matrices"][matrix_name]

    input_vector = _prepare_input_vector(
        request, matrix.config.rows, np.dtype(matrix._numpy_dtype)
    )
    result_vector = await matrix.rmatvec(input_vector)

    return VectorResponse(result=result_vector.tolist())


@app.post(
    "/api/v1/matmul/{matrix_a_name}/{matrix_b_name}",
    status_code=status.HTTP_200_OK,
    summary="Matrix-Matrix Multiplication",
)
async def perform_matmul(
    matrix_a_name: str, matrix_b_name: str, request: MatMulRequest
) -> dict[str, str]:
    """
    Performs matrix-matrix multiplication (C = A * B).
    """
    logger.info(
        f"Received matmul request for {matrix_a_name} * {matrix_b_name} -> "
        f"{request.output_uri}"
    )
    async with server_state["lock"].reader_lock:
        if matrix_a_name not in server_state["loaded_matrices"]:
            raise StreamMatException(
                ErrorCode.MATRIX_NOT_FOUND, f"Matrix '{matrix_a_name}' not found."
            )
        if matrix_b_name not in server_state["loaded_matrices"]:
            raise StreamMatException(
                ErrorCode.MATRIX_NOT_FOUND, f"Matrix '{matrix_b_name}' not found."
            )

        matrix_a = server_state["loaded_matrices"][matrix_a_name]
        matrix_b = server_state["loaded_matrices"][matrix_b_name]

    await matrix_a.matmul(matrix_b, request.output_uri)

    return {
        "message": (
            f"Matrix multiplication complete. Result stored at '{request.output_uri}'."
        )
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
