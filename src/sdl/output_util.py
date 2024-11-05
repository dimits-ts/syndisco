import matplotlib.pyplot as plt

import os
import textwrap


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


def save_plot(filename: str, dir_name: str = "output") -> None:
    """
    Saves a plot to the output directory.

    :param filename: The name of the file for the Figure.
    :type filename: str
    :param dir_name: The directory where the plot will be saved. Default is "output".
    :type dir_name: str
    """
    path = os.path.join(dir_name, filename)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.savefig(path, bbox_inches="tight")
    print(f"Figure saved to " + path)
