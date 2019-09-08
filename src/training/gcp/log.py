from google.cloud import logging


def setup_logging() -> None:
    # Instantiates a client
    client = logging.Client()
    # Connects the logger to the root logging handler; by default this captures
    # all logs at INFO level and higher
    client.setup_logging()
