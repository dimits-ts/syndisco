from sdl.postprocessing import postprocessing
import pandas as pd


CONV_PATH = "/home/dimits/Documents/research/synthetic_moderation_experiments/data/discussions_output"


df = postprocessing.import_conversations(CONV_PATH)
with pd.option_context(
    "display.max_rows",
    12,
    "display.max_columns",
    None,
    "display.precision",
    3,
):
    print(df)
