import sys

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

