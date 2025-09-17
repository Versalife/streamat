"""
This script is a wrapper to run the FastAPI server using 'uvicorn'.
"""

import uvicorn

from .server import app


def main() -> None:
    """Runs the FastAPI server using 'uvicorn'"""
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
