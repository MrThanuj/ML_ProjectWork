import logging
from pathlib import Path
from time import strftime

# Generating a unique log filename with current time
unique_log_name = f"applog_{strftime('%Y%m%d%H%M%S')}.txt"

# Specifying a directory within the current working path for logs
log_directory_path = Path(__file__).resolve().parent / "application_logs"

# Ensuring the log directory exists
log_directory_path.mkdir(parents=True, exist_ok=True)

# Full path for the log file
log_file_full_path = log_directory_path / unique_log_name

# Setting up the logging configuration
logging.basicConfig(
    filename=str(log_file_full_path),
    level=logging.DEBUG,
    format='%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

def log_info(message: str):
    """Function to log informational messages."""
    logging.info(message)

def log_error(message: str):
    """Function to log error messages."""
    logging.error(message)

def log_warning(message: str):
    """Function to log warnings."""
    logging.warning(message)