
import asyncio
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import tiledb
from loguru import logger

from .config import settings
from .conversion import MatrixFormat, convert_to_tiledb
from .models import DataType, StreamMatException, ErrorCode


class DataManager:
    """
    Manages the lifecycle of matrix data, including downloading, conversion,
    and caching.
    """

    def __init__(self, temp_dir: str = None, cache_dir: str = None):
        self.temp_dir = Path(temp_dir or tempfile.gettempdir())
        self.cache_dir = Path(cache_dir or settings.streammat_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._provision_locks = {}

    import asyncio
import functools
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable, Coroutine
from urllib.parse import urlparse

import aiohttp
import tiledb
from loguru import logger

from .config import settings
from .conversion import MatrixFormat, convert_to_tiledb
from .models import DataType, ErrorCode, StreamMatException


class DataManager:
    """
    Manages the lifecycle of matrix data, including downloading, conversion,
    and caching.
    """

    def __init__(self, temp_dir: str = None, cache_dir: str = None):
        self.temp_dir = Path(temp_dir or tempfile.gettempdir())
        self.cache_dir = Path(cache_dir or settings.streammat_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._provision_locks = {}

    async def provision_matrix(
        self, uri: str, matrix_name: str, progress_callback: Callable[[str], Coroutine] = None
    ) -> str:
        """
        Ensures a matrix is available in TileDB format, converting it if necessary.

        Args:
            uri: The URI of the matrix data (can be a local path or a URL).
            matrix_name: The name to be assigned to the matrix.
            progress_callback: An async function to call with progress updates.

        Returns:
            The local URI of the ready-to-use TileDB array.
        """
        if self._is_tiledb_array(uri):
            logger.info(f"URI '{uri}' is already a TileDB array. No conversion needed.")
            if progress_callback:
                await progress_callback("complete")
            return uri

        lock = self._get_provision_lock(uri)
        async with lock:
            cached_uri = self._get_cached_tiledb_uri(uri)
            if self._is_tiledb_array(cached_uri):
                logger.info(f"Found already cached TileDB array for URI '{uri}' at '{cached_uri}'.")
                if progress_callback:
                    await progress_callback("complete")
                return cached_uri

            logger.info(f"Provisioning matrix from source URI: {uri}")
            source_path, is_temporary = await self._fetch_source_file(uri, progress_callback)

            try:
                matrix_format = self._get_matrix_format(source_path)
                target_dtype = DataType.FLOAT64

                # Prepare the synchronous callback
                sync_callback = None
                if progress_callback:
                    loop = asyncio.get_running_loop()

                    def sync_progress_callback(written, total):
                        percent = (written / total) * 100
                        msg = f'{{"status": "converting", "progress": {percent:.2f}}}'
                        asyncio.run_coroutine_threadsafe(progress_callback(msg), loop)

                    sync_callback = sync_progress_callback

                conversion_task = functools.partial(
                    convert_to_tiledb,
                    input_path=source_path,
                    output_uri=cached_uri,
                    matrix_format=matrix_format,
                    target_dtype=target_dtype,
                    overwrite=True,
                    progress_callback=sync_callback,
                )
                
                await asyncio.to_thread(conversion_task)

                logger.info(f"Successfully converted '{uri}' to TileDB array at '{cached_uri}'.")
                if progress_callback:
                    await progress_callback('{{"status": "complete"}}')
                return cached_uri
            finally:
                if is_temporary:
                    self._cleanup_temp_file(source_path)

    def _get_provision_lock(self, uri: str) -> asyncio.Lock:
        """Returns a lock specific to a URI to prevent race conditions."""
        # Use a hash of the URI as the lock key
        lock_key = hashlib.sha256(uri.encode()).hexdigest()
        if lock_key not in self._provision_locks:
            self._provision_locks[lock_key] = asyncio.Lock()
        return self._provision_locks[lock_key]

    def _get_cached_tiledb_uri(self, source_uri: str) -> str:
        """Generates a deterministic cache path for a given source URI."""
        # Create a unique, filesystem-safe name from the URI
        uri_hash = hashlib.sha256(source_uri.encode()).hexdigest()
        return str(self.cache_dir / f"{uri_hash}.tiledb")

    def _is_tiledb_array(self, uri: str) -> bool:
        """Checks if a given URI points to a valid TileDB array."""
        if not Path(uri).exists():
            return False
        try:
            # The object_type check is a lightweight way to see if it's a TileDB resource
            return tiledb.object_type(uri) == "array"
        except tiledb.TileDBError:
            return False

    async def _fetch_source_file(self, uri: str, progress_callback: Callable[[str], Coroutine] = None) -> tuple[Path, bool]:
        """
        Fetches the source file, downloading it if it's remote.

        Returns:
            A tuple of (Path to the local file, boolean indicating if it's temporary).
        """
        if self._is_remote(uri):
            # It's a remote file, so we need to download it.
            logger.info(f"URI is remote. Downloading from {uri}...")
            file_path = await self._download_file(uri, progress_callback)
            return file_path, True
        else:
            # It's a local file.
            local_path = Path(uri)
            if not local_path.exists():
                raise StreamMatException(ErrorCode.INVALID_REQUEST, f"Local file not found: {uri}")
            logger.info(f"URI is a local file: {local_path}")
            return local_path, False

    def _is_remote(self, uri: str) -> bool:
        """Checks if a URI has a scheme that suggests it's a remote URL."""
        return urlparse(uri).scheme in ["http", "https"]

    async def _download_file(self, url: str, progress_callback: Callable[[str], Coroutine] = None) -> Path:
        """Downloads a file from a URL to the temporary directory."""
        # It's good practice to check headers first to avoid downloading huge, unsupported files.
        async with aiohttp.ClientSession() as session:
            try:
                async with session.head(url, allow_redirects=True) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    total_size = int(response.headers.get("Content-Length", 0))
                    logger.info(f"Remote file content type: {content_type}, size: {total_size} bytes")

            except aiohttp.ClientError as e:
                raise StreamMatException(
                    ErrorCode.NETWORK_ERROR, f"Failed to retrieve headers from {url}: {e}"
                )

            # Proceed with downloading the file
            temp_file_path = self.temp_dir / os.path.basename(urlparse(url).path)

            try:
                downloaded_size = 0
                async with session.get(url, allow_redirects=True) as response:
                    response.raise_for_status()
                    with open(temp_file_path, "wb") as f:
                        while True:
                            chunk = await response.content.read(1024 * 1024)  # Read in 1MB chunks
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if progress_callback and total_size > 0:
                                percent = (downloaded_size / total_size) * 100
                                msg = f'{{"status": "downloading", "progress": {percent:.2f}}}'
                                await progress_callback(msg)

                logger.info(f"Successfully downloaded file to {temp_file_path}")
                return temp_file_path
            except aiohttp.ClientError as e:
                raise StreamMatException(
                    ErrorCode.NETWORK_ERROR, f"Failed to download file from {url}: {e}"
                )

    def _get_matrix_format(self, path: Path) -> MatrixFormat:
        """Infers the matrix format from the file extension."""
        suffix = path.suffix
        if suffix == ".gz":
            suffix = path.with_suffix("").suffix

        if suffix in [".mtx", ".mm"]:
            return MatrixFormat.MATRIX_MARKET
        elif suffix in [".hb", ".rua"]:
            return MatrixFormat.HARWELL_BOEING
        elif suffix == ".gct":
            return MatrixFormat.GCT
        else:
            raise StreamMatException(
                ErrorCode.UNSUPPORTED_FORMAT,
                f"Could not infer format from file extension '{suffix}'."
            )

    def _cleanup_temp_file(self, path: Path):
        """Deletes a file from the temporary directory."""
        try:
            logger.info(f"Cleaning up temporary file: {path}")
            os.remove(path)
        except OSError as e:
            logger.warning(f"Failed to clean up temporary file {path}: {e}")

