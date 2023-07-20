import logging
import sys


def setup_logging(level, console_log, log_file, name):
    """Setup logging based on specified inputs

    Parameters
    ----------
    level : str
        Level of logging
    console_log : bool
        Whether to print logs to console or not
    log_file : str
        Path of the log file
    name : str
        Name of the logger

    Returns
    -------
    logging.Logger
        Logger with specified handlers and formatting.
    """
    handlers = [logging.FileHandler(log_file)]
    if not console_log:
        handlers.append(logging.StreamHandler(sys.stdout))
        logging.basicConfig(level=level.upper(), handlers=handlers)
    else:
        logging.basicConfig(level=level.upper(), handlers=handlers)
    return logging.getLogger(name)


if __name__ == "__main__":
    pass
