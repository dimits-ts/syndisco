import matplotlib.pyplot as plt

import os
import datetime
import textwrap


def ensure_parent_directories_exist(output_path: str) -> None:
    """
    Create all parent directories if they do not exist.
    :param output_path: the path for which parent dirs will be generated
    """
    # Extract the directory path from the given output path
    directory = os.path.dirname(output_path)

    # Create all parent directories if they do not exist
    if directory:
        os.makedirs(directory, exist_ok=True)


def generate_datetime_filename(
        output_dir: str = None, timestamp_format: str = "%y-%m-%d-%H-%M", file_ending: str = ""
) -> str:
    """
    Generate a filename based on the current date and time.

    :param output_dir: The path to the generated file, defaults to None
    :type output_dir: str, optional
    :param timestamp_format: strftime format, defaults to "%y-%m-%d-%H-%M"
    :type timestamp_format: str, optional
    :param file_ending: The ending of the file (e.g '.json')
    :type file_ending: str
    :return: the full path for the generated file
    :rtype: str
    """
    datetime_name = datetime.datetime.now().strftime(timestamp_format) + file_ending

    if output_dir is None:
        return datetime_name
    else:
        return os.path.join(output_dir, datetime_name)


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
        formatted_res = f"<{username} said nothing>"

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
