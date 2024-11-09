# import logger.py from src folder
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Function to handle exceptions
import logging


# Function to extract detailed error information
def error_handling_details(error):
    """
    Extracts details about where an error occurred (file name, line number) and formats it into a message.
    
    Args:
        error (Exception): The original exception object.
    
    Returns:
        str: A formatted error message with file name, line number, and error message.
    """
    _, _, exc_tb = sys.exc_info()  # Get traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file where the error occurred
    line_number = exc_tb.tb_lineno  # Get the line number where the error occurred
    error_message = f"Error in python script [{file_name}] at line [{line_number}]: {str(error)}"
    
    return error_message

# Custom Exception Handler Class
class customExceptionHandler(Exception):
    def __init__(self, error_message):
        """
        Initializes the custom exception handler with a detailed error message.
        
        Args:
            error_message (str): The original error message.
        """
        super().__init__(error_message)
        self.error_message = error_handling_details(error_message)
    
    def __str__(self):
        return self.error_message

# Main block for testing custom exception handling
if __name__ == "__main__":
    try:
        # Example of an operation that causes an exception (division by zero)
        a = 1 / 0
    except Exception as e:
        # Raise custom exception with detailed information
        logging.error("Divide by zero exception!!!", exc_info=True)
        raise customExceptionHandler(e)  # Use from e for proper exception chaining