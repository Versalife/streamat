"""
Defines the data models for the application using Pydantic.

This ensures strong typing, validation, and clear data contracts, especially
for API request and response bodies.
"""

import enum

from pydantic import BaseModel, Field


class DataType(str, enum.Enum):
    """Enumeration for matrix data types, for consistency in metadata."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    # Add other types as needed


class ErrorCode(str, enum.Enum):
    """Enumeration for application-specific error codes."""

    MATRIX_NOT_FOUND = "MATRIX_NOT_FOUND"
    TILEDB_ERROR = "TILEDB_ERROR"
    INVALID_TILEDB_SCHEMA = "INVALID_TILEDB_SCHEMA"
    METADATA_ERROR = "METADATA_ERROR"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    INVALID_REQUEST = "INVALID_REQUEST"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    DATA_TYPE_MISMATCH = "DATA_TYPE_MISMATCH"
    UNSUPPORTED_DATATYPE = "UNSUPPORTED_DATATYPE"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    NETWORK_ERROR = "NETWORK_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class StreamMatConfig(BaseModel):
    """
    Data model for metadata stored within a TileDB array.
    This replaces the C++ StreamMatHeader struct.
    """

    rows: int = Field(..., gt=0, description="Number of rows in the matrix")
    cols: int = Field(..., gt=0, description="Number of columns in the matrix")
    nnz: int = Field(..., ge=0, description="Number of non-zero elements")
    dtype: DataType = Field(..., description="Data type of the matrix values")
    version: int = Field(1, description="StreamMat version used to create the matrix")


class MatrixInfo(BaseModel):
    """Data model describing a single loaded matrix for status responses."""

    uri: str
    config: StreamMatConfig
    operations: dict[str, "OperationInfo"]


class OperationInfo(BaseModel):
    """Describes the expected shapes for a matrix operation."""

    input_shape: list[int]
    output_shape: list[int]


class ServerStatus(BaseModel):
    """Data model for the GET /status API endpoint response."""

    server_status: str
    loaded_matrices: dict[str, MatrixInfo]


class SparseVector(BaseModel):
    """Represents a sparse vector using indices and values."""

    indices: list[int]
    values: list[float]


class VectorRequest(BaseModel):
    """Data model for API requests requiring a vector (e.g., matvec)."""

    vector: list[float] | SparseVector


class VectorResponse(BaseModel):
    """Data model for API responses returning a vector."""

    result: list[float]


class LoadMatrixRequest(BaseModel):
    """Data model for the PUT /matrix/{matrix_name} API request."""

    uri: str = Field(
        ...,
        description="The URI of the TileDB array to load (e.g., path or tiledb://namespace/array)",
    )


class MatMulRequest(BaseModel):
    """Data model for the POST /matmul/{matrix_a_name}/{matrix_b_name} API request."""

    output_uri: str = Field(
        ..., description="The URI for the output TileDB array to be created."
    )


class ErrorDetail(BaseModel):
    """Detailed error model for API error responses."""

    code: ErrorCode
    message: str


class ErrorResponse(BaseModel):
    """Top-level model for API error responses."""

    error: ErrorDetail


class StreamMatException(Exception):
    """Base exception for the StreamMat application."""

    def __init__(self, code: ErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{self.code.value}] {self.message}")
