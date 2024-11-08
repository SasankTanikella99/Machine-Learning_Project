
import logging
import os
from datetime import datetime

# This code snippet is setting up a logging configuration in Python. 

LOG_FILE= f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

# This code snippet is setting up the logging configuration in Python using the `basicConfig` method
# from the `logging` module. Here's what each parameter in the `basicConfig` function call is doing:
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# The `if __name__ == "__main__":` block in Python is a common idiom used to check if the current
# script is being run directly by the Python interpreter as the main program.
if __name__ == "__main__":
    logging.info("Starting")