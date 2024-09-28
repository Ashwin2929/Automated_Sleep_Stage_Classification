#!/usr/bin/python36

"""
Created on Thu Jun 13 20:30:34 2024

@author: ashwin
"""


import logging
import os

# Ensure output directory exists
output_directory = './output'
os.makedirs(output_directory, exist_ok=True)

# Configure logging to overwrite the log file
logging.basicConfig(filename=os.path.join(output_directory, 'log.txt'), 
                    level=logging.INFO, 
                    format='%(message)s', 
                    filemode='w')

def log_print(message):
    """
    Log a message to both the console and the log file.

    Parameters:
        - message: string. The message to log.

    Returns:
        - None
    """
    
    print(message)  # Print to console
    logging.info(message)  # Log to file

log_print("----------------------------------------------------------------------------------------")
log_print("This is the log for sleep.")
log_print("----------------------------------------------------------------------------------------")

# Ensure logging handlers are properly flushed
logging.shutdown()