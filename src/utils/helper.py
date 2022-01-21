import logging

def get_logger():
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger