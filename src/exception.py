import sys
from src.logger import custom_log  
def format_error_details(error, detail: sys):
    """
    Extracts and formats error details for improved error reporting.
    """
    traceback_info = detail.exc_info()[2]
    script_name = traceback_info.tb_frame.f_code.co_filename
    line_number = traceback_info.tb_lineno
    formatted_message = f"Error encountered in script [{script_name}] at line [{line_number}]: {str(error)}"
    return formatted_message

class CustomException(Exception):
    """
    A custom exception class for handling operational issues within the application.
    """
    def __init__(self, message, detail: sys):
        super().__init__(message)
        self.detailed_error_message = format_error_details(message, detail)

    def __str__(self):
        return self.detailed_error_message
