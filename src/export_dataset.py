import pandas as pd

import logging
import argparse
import yaml

from sdl.postprocessing import postprocessing
from sdl.util.logging_util import logging_setup


def main():
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Generate annotation files using configuration files."
    )
    parser.add_argument(
        "--config_file", required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config_file, "r") as file:
        config_data = yaml.safe_load(file)

    logging_config = config_data["logging"]

    logging_setup(
        print_to_terminal=logging_config["print_to_terminal"],
        write_to_file=logging_config["write_to_file"],
        logs_dir=logging_config["logs_dir"],
        level=logging_config["level"]
    )

    export_data = config_data["export"]
    conv_dir = export_data["conversations_dir"]
    annot_dir = export_data["annotations_dir"]
    output_path = export_data["output_path"]
    include_sdb = export_data["include_annotator_sdbs"]

    if not include_sdb:
        logger.warning("Not including annotator SDBs to output.")

    conv_df = postprocessing.import_conversations(conv_dir)
    conv_df = conv_df.rename({"id": "conv_id"}, axis=1)
    annot_df = postprocessing.import_annotations(annot_dir, include_sdb)


    full_df = pd.merge(
        left=conv_df,
        right=annot_df,
        on=["conv_id", "message"],
        how="left",
        suffixes=["_conv", "_annot"] # type: ignore
    )
    del full_df["index_annot"]
    del full_df["index_conv"]

    full_df.to_csv(path_or_buf=output_path, encoding="utf8")
    logger.info("Dataset exported to: " + output_path)


if __name__ == "__main__":
    main()
