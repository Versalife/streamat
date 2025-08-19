"""
Contains the core StreamMatrix class for performing matrix operations.
"""
import asyncio
from typing import Self, Type

import numpy as np
import tiledb
from loguru import logger

from .models import DataType, ErrorCode, StreamMatConfig, StreamMatException

# Constants for TileDB schema and metadata
DIM_ROW_NAME = "dim_row"
DIM_COL_NAME = "dim_col"
ATTR_VALUE_NAME = "attr_value"


class StreamMatrix:
    """
    A class representing a sparse matrix backed by a TileDB array,
    providing high-performance matrix-vector operations.
    """

    def __init__(self: Self, uri: str, ctx: tiledb.Ctx) -> None:
        """
        Initializes the StreamMatrix by opening the TileDB array and validating it.

        Args:
            uri: The URI of the TileDB sparse array.
            ctx: A TileDB context object.

        Raises:
            StreamMatException: If the array cannot be opened, is not a valid sparse
                                matrix schema, or metadata is missing/invalid.
        """
        self.uri: str = uri
        self._ctx: tiledb.Ctx = ctx
        self.config: StreamMatConfig
        self._numpy_dtype: Type[np.number]
        logger.info(f"Initializing StreamMatrix for array: '{self.uri}'")

        try:
            # Validate and load the array schema and metadata
            self._array: tiledb.Array = tiledb.Array(self.uri, mode='r', ctx=self._ctx)
            self._load_and_validate_meta()
            logger.success(f"Successfully validated and loaded metadata for '{self.uri}'.")
        except tiledb.TileDBError as e:
            logger.error(f"TileDB error while opening array '{self.uri}': {e}")
            raise StreamMatException(
                ErrorCode.TILEDB_ERROR,
                f"Failed to open or validate TileDB array '{self.uri}': {e}"
            ) from e

    def _load_and_validate_meta(self: Self) -> None:
        """Loads and validates the array schema and metadata."""
        logger.debug(f"Loading and validating metadata for '{self.uri}'...")
        schema = self._array.schema
        if not schema.sparse or schema.ndim != 2:
            raise StreamMatException(
                ErrorCode.INVALID_TILEDB_SCHEMA,
                f"Array '{self.uri}' must be a 2D sparse array."
            )
        if not schema.has_attr(ATTR_VALUE_NAME):
            raise StreamMatException(
                ErrorCode.INVALID_TILEDB_SCHEMA,
                f"Array '{self.uri}' must have an attribute named '{ATTR_VALUE_NAME}'."
            )

        # Load metadata into the Pydantic model for validation
        meta_dict = {k: v for k, v in self._array.meta.items()}
        self.config = StreamMatConfig.model_validate(meta_dict)

        # Map DataType enum to numpy dtype
        dtype_map = {
            DataType.FLOAT32: np.float32,
            DataType.FLOAT64: np.float64,
            DataType.INT32: np.int32,
            DataType.INT64: np.int64,
        }
        if self.config.dtype not in dtype_map:
            raise StreamMatException(
                ErrorCode.UNSUPPORTED_DATATYPE,
                f"Unsupported dtype '{self.config.dtype}' in matrix config."
            )
        self._numpy_dtype = dtype_map[self.config.dtype]

        # Verify that the attribute's dtype matches the metadata
        if self._array.schema.attr(ATTR_VALUE_NAME).dtype != self._numpy_dtype:
            raise StreamMatException(
                ErrorCode.DATA_TYPE_MISMATCH,
                f"Attribute dtype mismatch. Schema has '{self._array.schema.attr(ATTR_VALUE_NAME).dtype}', "
                f"but metadata specifies '{self._numpy_dtype}'."
            )
        logger.debug("Metadata and schema validation passed.")

    async def matvec(self: Self, vector: np.ndarray) -> np.ndarray:
        """
        Performs matrix-vector multiplication (y = A * x).

        Args:
            vector: A 1D numpy array representing the input vector x.

        Returns:
            A 1D numpy array representing the resulting vector y.
        
        Raises:
            StreamMatException: If vector dimensions do not match.
        """
        if vector.shape != (self.config.cols,):
            raise StreamMatException(
                ErrorCode.DIMENSION_MISMATCH,
                f"Input vector has shape {vector.shape}, but matrix has {self.config.cols} columns."
            )
        if self.config.nnz == 0:
            logger.warning("Performing matvec on a matrix with zero non-zero elements.")
            return np.zeros(self.config.rows, dtype=self._numpy_dtype)

        logger.info(f"Starting matvec operation for {self.config.rows} rows.")
        # Create an async task for each row's dot product calculation
        tasks = [self._get_row_dot_product(r, vector) for r in range(self.config.rows)]

        # Execute all tasks concurrently
        result_vector = await asyncio.gather(*tasks)

        logger.success("matvec operation completed successfully.")
        return np.array(result_vector, dtype=self._numpy_dtype)

    async def _get_row_dot_product(self: Self, row_index: int, vector: np.ndarray) -> np.number:
        """Helper to compute the dot product for a single row concurrently."""
        try:
            # Query one row: A[row_index, :]
            row_data = self._array.multi_index[row_index:row_index, :]

            # Get column indices and values for non-zero elements
            col_indices = row_data[DIM_COL_NAME]
            values = row_data[ATTR_VALUE_NAME]

            if col_indices.size == 0:
                return self._numpy_dtype(0)

            # Use numpy's optimized dot product
            return values.dot(vector[col_indices])
        except Exception:
            # Use logger.exception to automatically include traceback info
            logger.exception(f"An error occurred while processing row {row_index}.")
            return self._numpy_dtype(0)

    async def rmatvec(self: Self, vector: np.ndarray) -> np.ndarray:
        """
        Performs adjoint matrix-vector multiplication (y = A.T * x) by
        concurrently calculating the dot product of each column with the input vector.

        Args:
            vector: A 1D numpy array representing the input vector x.

        Returns:
            A 1D numpy array representing the resulting vector y.

        Raises:
            StreamMatException: If vector dimensions do not match.
        """
        if vector.shape != (self.config.rows,):
            raise StreamMatException(
                ErrorCode.DIMENSION_MISMATCH,
                f"Input vector has shape {vector.shape}, but matrix has {self.config.rows} rows."
            )
        if self.config.nnz == 0:
            logger.warning("Performing rmatvec on a matrix with zero non-zero elements.")
            return np.zeros(self.config.cols, dtype=self._numpy_dtype)

        logger.info(f"Starting rmatvec operation for {self.config.cols} columns.")
        # Create an async task for each column's dot product calculation
        tasks = [self._get_col_dot_product(c, vector) for c in range(self.config.cols)]

        # Execute all tasks concurrently
        result_vector = await asyncio.gather(*tasks)

        logger.success("rmatvec operation completed successfully.")
        return np.array(result_vector, dtype=self._numpy_dtype)

    async def _get_col_dot_product(self: Self, col_index: int, vector: np.ndarray) -> np.number:
        """Helper to compute the dot product for a single column concurrently."""
        try:
            # Query one column: A[:, col_index]
            col_data = self._array.multi_index[:, col_index:col_index]

            # Get row indices and values for non-zero elements in this column
            row_indices = col_data[DIM_ROW_NAME]
            values = col_data[ATTR_VALUE_NAME]

            if row_indices.size == 0:
                return self._numpy_dtype(0)

            # Use numpy's optimized dot product: sum(values[k] * vector[row_indices[k]])
            return values.dot(vector[row_indices])
        except Exception:
            logger.exception(f"An error occurred while processing column {col_index}.")
            return self._numpy_dtype(0)
