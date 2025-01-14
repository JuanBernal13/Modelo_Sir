# utils/logger.py

import logging

def get_logger():
    logger = logging.getLogger("EpidemicSimulation")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
