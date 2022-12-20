from langchain.logging.base import BaseLogger
from langchain.logging.sqlite import SqliteLogger


def get_logger() -> BaseLogger:
    return SqliteLogger()
