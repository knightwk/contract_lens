from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger


class AppError(Exception):
    def __init__(self, message: str, code: int = 400):
        super().__init__(message, code)
        self.message = message
        self.code = code


class PydanticValidationError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code=422)


class FileHandlingError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code=500)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request, exc: AppError):
        logger.error(f"Pydantic validation error: {exc.message}")
        return JSONResponse(
            status_code=exc.code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    @app.exception_handler(PydanticValidationError)
    async def handle_validation_error(request, exc: PydanticValidationError):
        logger.error(f"Pydantic validation error: {exc.message}")
        return JSONResponse(
            status_code=exc.code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    @app.exception_handler(FileHandlingError)
    async def handle_file_handling_error(request, exc: FileHandlingError):
        logger.error(f"File handling error: {exc.message}")
        return JSONResponse(
            status_code=exc.code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    @app.exception_handler(Exception)
    async def handle_generic_error(request, exc: Exception):
        logger.exception(f"Pydantic validation error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": "服务器内部错误", "data": None},
        )
