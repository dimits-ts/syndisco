import textwrap
import logging
from time import time
from functools import wraps
from pathlib import Path
from typing import Callable, Any


logger = logging.getLogger(Path(__file__).name)

def format_chat_message(username: str, message: str) -> str:
    """
    Create a prompt-friendly/console-friendly string representing a message
    made by a user.

    :param username: the name of the user who made the post
    :type username: str
    :param message: the message that was posted
    :type message: str
    :return: a formatted string containing both username and his message
    :rtype: str
    """
    if len(message.strip()) != 0:
        # append name of actor to his response
        # "user x posted" important for the model to not confuse it with the prompt
        wrapped_res = textwrap.fill(message, 70)
        formatted_res = f"User {username} posted:\n{wrapped_res}"
    else:
        formatted_res = ""

    return formatted_res


# from https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(f: Callable) -> Any:
    """
    Decorator which logs the execution time of a function.

    :param f: the function to be timed
    :type f: Function
    :return: the result of the function
    :rtype: _type_
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        exec_time_mins = (te-ts)/60
        logger.info(f"Procedure {f.__name__} executed in {exec_time_mins:2.4f} minutes.")
        return result
    return wrap
