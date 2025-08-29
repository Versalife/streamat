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

    async def matvec(self: Self, vector: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
        """
        Performs matrix-vector multiplication (y = A * x).

        Args:
            vector: A 1D numpy array representing the input vector x.
            chunk_size: The number of rows to process in each concurrent chunk.

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

        logger.info(f"Starting matvec operation for {self.config.rows} rows with chunk size {chunk_size}.")
        
        result_vector = np.zeros(self.config.rows, dtype=self._numpy_dtype)
        
        tasks = []
        for r_start in range(0, self.config.rows, chunk_size):
            r_end = min(r_start + chunk_size, self.config.rows)
            tasks.append(self._get_row_chunk_dot_product(r_start, r_end, vector))

        try:
            chunk_results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.exception(f"An error occurred during matvec for matrix {self.uri}")
            raise StreamMatException(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"A processing error occurred in matvec: {e}"
            ) from e

        for r_start, chunk_result in zip(range(0, self.config.rows, chunk_size), chunk_results):
            r_end = r_start + len(chunk_result)
            result_vector[r_start:r_end] = chunk_result

        logger.success("matvec operation completed successfully.")
        return result_vector

    async def _get_row_chunk_dot_product(self: Self, r_start: int, r_end: int, vector: np.ndarray) -> np.ndarray:
        """Helper to compute the dot product for a chunk of rows concurrently."""
        row_data = self._array.multi_index[r_start:r_end-1, :]
        
        row_indices = row_data[DIM_ROW_NAME]
        col_indices = row_data[DIM_COL_NAME]
        values = row_data[ATTR_VALUE_NAME]

        if values.size == 0:
            return np.zeros(r_end - r_start, dtype=self._numpy_dtype)

        # Create a sparse matrix from the chunk data
        from scipy.sparse import coo_matrix
        scipy_sparse_matrix = coo_matrix((values, (row_indices - r_start, col_indices)), shape=(r_end - r_start, self.config.cols))

        return scipy_sparse_matrix.dot(vector)

    async def rmatvec(self: Self, vector: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
        """
        Performs adjoint matrix-vector multiplication (y = A.T * x).

        Args:
            vector: A 1D numpy array representing the input vector x.
            chunk_size: The number of columns to process in each concurrent chunk.

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

        logger.info(f"Starting rmatvec operation for {self.config.cols} columns with chunk size {chunk_size}.")
        
        result_vector = np.zeros(self.config.cols, dtype=self._numpy_dtype)

        tasks = []
        for c_start in range(0, self.config.cols, chunk_size):
            c_end = min(c_start + chunk_size, self.config.cols)
            tasks.append(self._get_col_chunk_dot_product(c_start, c_end, vector))

        try:
            chunk_results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.exception(f"An error occurred during rmatvec for matrix {self.uri}")
            raise StreamMatException(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"A processing error occurred in rmatvec: {e}"
            ) from e

        for c_start, chunk_result in zip(range(0, self.config.cols, chunk_size), chunk_results):
            c_end = c_start + len(chunk_result)
            result_vector[c_start:c_end] = chunk_result

        logger.success("rmatvec operation completed successfully.")
        return result_vector

    async def _get_col_chunk_dot_product(self: Self, c_start: int, c_end: int, vector: np.ndarray) -> np.ndarray:
        """Helper to compute the dot product for a single column concurrently."""
        col_data = self._array.multi_index[:, c_start:c_end-1]

        row_indices = col_data[DIM_ROW_NAME]
        col_indices = col_data[DIM_COL_NAME]
        values = col_data[ATTR_VALUE_NAME]

        if values.size == 0:
            return np.zeros(c_end - c_start, dtype=self._numpy_dtype)

        # Create a sparse matrix from the chunk data
        from scipy.sparse import coo_matrix
        scipy_sparse_matrix = coo_matrix((values, (row_indices, col_indices - c_start)), shape=(self.config.rows, c_end - c_start))

        return scipy_sparse_matrix.T.dot(vector)

    async def matmul(self: Self, other: Self, output_uri: str, chunk_size: int = 100) -> None:
        """
        Performs matrix-matrix multiplication (C = A * B).

        Args:
            other: The right-hand side matrix (B).
            output_uri: The URI for the output TileDB array (C).
            chunk_size: The number of columns of B to process in each chunk.
        """
        if self.config.cols != other.config.rows:
            raise StreamMatException(
                ErrorCode.DIMENSION_MISMATCH,
                f"Matrix A has {self.config.cols} columns, but matrix B has {other.config.rows} rows."
            )

        logger.info(f"Starting matmul operation: {self.uri} * {other.uri} -> {output_uri}")

        # Create the output array
        output_rows, output_cols = self.config.rows, other.config.cols
        domain = tiledb.Domain(
            tiledb.Dim(name=DIM_ROW_NAME, domain=(0, output_rows - 1), tile=min(output_rows, 4096), dtype=np.uint64),
            tiledb.Dim(name=DIM_COL_NAME, domain=(0, output_cols - 1), tile=min(output_cols, 4096), dtype=np.uint64),
        )
        schema = tiledb.ArraySchema(
            domain=domain,
            sparse=True,
            attrs=[tiledb.Attr(name=ATTR_VALUE_NAME, dtype=self._numpy_dtype)]
        )
        tiledb.Array.create(output_uri, schema, ctx=self._ctx)

        # Process B in column chunks
        for c_start in range(0, other.config.cols, chunk_size):
            c_end = min(c_start + chunk_size, other.config.cols)
            
            tasks = [self.matvec(other._get_col(c)) for c in range(c_start, c_end)]
            
            try:
                result_cols = await asyncio.gather(*tasks)
            except Exception as e:
                logger.exception(f"An error occurred during matmul for matrix {self.uri}")
                raise StreamMatException(
                    ErrorCode.INTERNAL_SERVER_ERROR,
                    f"A processing error occurred in matmul: {e}"
                ) from e

            # Write the resulting columns to the output array
            with tiledb.open(output_uri, 'w', ctx=self._ctx) as C:
                for i, col_vec in enumerate(result_cols):
                    col_index = c_start + i
                    row_indices = np.nonzero(col_vec)[0]
                    values = col_vec[row_indices]
                    col_indices = np.full_like(row_indices, col_index)
                    C[row_indices, col_indices] = values
        
        logger.success("matmul operation completed successfully.")

    def _get_col(self: Self, col_index: int) -> np.ndarray:
        """Helper to get a single column as a dense numpy array."""
        col_data = self._array.multi_index[:, col_index:col_index]
        
        row_indices = col_data[DIM_ROW_NAME]
        values = col_data[ATTR_VALUE_NAME]

        col_vector = np.zeros(self.config.rows, dtype=self._numpy_dtype)
        col_vector[row_indices] = values
        return col_vector
