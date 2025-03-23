from logging_config import setup_logging
import argparse

if __name__ == "__main__":
    logger = setup_logging()

    # TODO: add argparse to get input file

    logger.info('Starting geometry calculation.')
    # TODO: add loading geometry file
