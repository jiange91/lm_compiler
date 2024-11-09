import logging
import warnings
import optuna

def _create_default_formatter(log_level) -> logging.Formatter:
    """Create a default formatter of log messages.

    This function is not supposed to be directly accessed by library users.
    """
    if log_level == 'DEBUG':
        header = "[%(levelname)1s %(asctime)s - %(pathname)s:%(lineno)d]"
    else:
        header = "[%(levelname)1s %(asctime)s]"
    message = "%(message)s"
    formatter = logging.Formatter(
        fmt=f"{header} {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return formatter

def _configure_logger(log_level):
    # config root logger
    handler = logging.StreamHandler()
    handler.setFormatter(_create_default_formatter(log_level))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('absl').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)

    warnings.filterwarnings("ignore", module="pydantic")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning, module="optuna")
    warnings.filterwarnings("ignore", category=FutureWarning)


    