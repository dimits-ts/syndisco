"""
SynDisco: Automated experiment creation and execution using only LLM agents
Copyright (C) 2025 Dimitris Tsirmpas

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

You may contact the author at tsirbasdim@gmail.com
"""

from pathlib import Path
import logging

from ..backend import model


logger = logging.getLogger(Path(__file__).name)


class ModelManager:
    """
    A Singleton class initializing and managing access to a single,
    unique instance of a model.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int,
        disallowed_strings: list[str],
        model_pseudoname: str | None = None,
    ):
        """
        Initialize the manager without loading the model to the runtime.

        :param yaml_data: the experiment configuration
        :type yaml_data: dict
        """
        self.model = None
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.disallowed_strings = disallowed_strings
        self.model_pseudoname = (
            model_path if model_pseudoname is None else model_pseudoname
        )

    def get(self) -> model.BaseModel:
        """
        Get a reference to the protected model instance.
        First invocation loads the instance to runtime.

        :raises NotImplementedError: if an incompatible library_type
         is given in the yaml_data of the constructor
        :return: The initialized model instance.
        :rtype: model.Model
        """
        if self.model is None:
            logger.info("Loading model...")
            self.model = self._initialize_model()
            logger.info("Model loaded.")
        else:
            logger.info("Using already loaded model...")

        return self.model

    def _initialize_model(self) -> model.BaseModel:
        """
        Initialize a new LLM model wrapper instance.

        :raises NotImplementedError: if an incompatible library_type is given
        :return: an initialized, loaded LLM model wrapper
        :rtype: model.Model
        """
        # Extract values from the config

        return model.TransformersModel(
            model_path=self.model_path,
            name=self.model_pseudoname,
            max_out_tokens=self.max_new_tokens,
            remove_string_list=self.disallowed_strings,
        )
