# Gemini Agent Best Practices for the StreamMat Project

This document provides guidelines for Gemini agents working on the `streammat` project. Adhering to these practices will ensure code quality, consistency, and maintainability.

## 1. Project Overview

StreamMat is a Python library designed for high-performance, out-of-core sparse matrix operations. It uses TileDB as its storage backend to efficiently handle matrices that are too large to fit into memory. The library provides a FastAPI-based web server to expose matrix operations as a RESTful API.

## 2. Development Setup

The project uses `uv` for dependency management and as a project runner.

1.  **Install `uv`**:
    ```bash
    pip install uv
    ```

2.  **Create a virtual environment and install dependencies**:
    ```bash
    uv venv
    uv sync --all-extras
    ```

## 3. Running Checks

Before committing any changes, it is crucial to run all the checks to ensure that the code is correct and follows the project's standards.

-   **Run tests with `pytest`**:
    ```bash
    uv run pytest
    ```

-   **Run type checks with `mypy`**:
    ```bash
    uv run mypy .
    ```

-   **Run lints with `ruff`**:
    ```bash
    uv run ruff check .
    ```

-   **Run all checks together**:
    ```bash
    uv run pytest && uv run mypy . && uv run ruff check .
    ```

## 4. Code Style and Conventions

-   **Follow existing code style**: When modifying a file, mimic the style and structure of the surrounding code.
-   **Linting and Formatting**: The project uses `ruff` for linting. Use `ruff check . --fix` to automatically fix any linting issues.
-   **Type Hinting**: All new functions and methods should have type hints. The project uses `mypy` to enforce this.

## 5. CI/CD Pipeline

The CI/CD pipeline is defined in `.github/workflows/ci.yml`. It is triggered on every push and pull request to the `main` branch and runs all the checks to ensure the integrity of the codebase.

## 6. Dependencies

All project and development dependencies are managed with `uv` and are defined in the `pyproject.toml` file. Do not use `pip` to install dependencies directly.

## 7. Worklogs

Worklogs are maintained in the `worklogs` directory. For any significant work done on the project, create a new markdown file with the date as the filename (e.g., `YYYY-MM-DD.worklog.md`) and document the changes made.

## 8. Project Plan

The `PLAN.md` file contains the development plan for the project. It outlines the future work and the direction of the project. Before starting any new work, consult this file to understand the project's roadmap and priorities.