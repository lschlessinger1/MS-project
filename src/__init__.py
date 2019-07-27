# See https://docs.python-guide.org/writing/logging/#logging-in-a-library
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
