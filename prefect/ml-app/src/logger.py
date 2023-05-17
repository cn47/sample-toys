from logging import getLogger
from logging.config import dictConfig
from pathlib import Path

### logger
LOGGER_CONF = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "basic": {
            "format": "%(asctime)s  [%(name)s][%(levelname)s]  %(message)s  %(filename)s",
            # 'format': '%(asctime)s  [%(name)s][%(levelname)s]  %(message)s  (%(module)s:%(filename)s:%(funcName)s:%(lineno)d)',
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "basic",
            "filename": "__filename",
            "when": "MIDNIGHT",
            "interval": 1,
            "backupCount": 100,
        },
        "error": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "ERROR",
            "formatter": "basic",
            "filename": "__filename",
            "when": "MIDNIGHT",
            "interval": 1,
            "backupCount": 100,
        },
    },
    "loggers": {"APP_NAME": {"level": "INFO", "handlers": ["console", "file", "error"], "propagate": False}},
    "root": {"level": "INFO", "handlers": ["console", "file", "error"]},
}


def get_logger(app_name: str = "PythonApp", log_file: str = None):
    if log_file is None:
        log_file = Path(__file__).parents[1] / "log" / "app.log"
    log_file = Path(log_file)
    log_file.resolve().parent.mkdir(parents=True, exist_ok=True)

    filename = str(log_file)
    filename_err = str(log_file.parent / f"err_{log_file.name}")
    LOGGER_CONF["handlers"]["file"].update({"filename": filename})
    LOGGER_CONF["handlers"]["error"].update({"filename": filename_err})

    dictConfig(LOGGER_CONF)
    return getLogger(app_name)
