
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


"""
Holds a utility class which helps manage initialization, loading and unloading of models on demand.
"""

from pathlib import Path
import logging

from ..backend import model


logger = logging.getLogger(Path(__file__).name)


class ModelManager:
    """
    A Factory and Singleton class initializing and managing access to a single,
    unique instance of a model.
    """

    def __init__(self, yaml_data: dict):
        """
        Initialize the manager without loading the model to the runtime.

        :param yaml_data: the experiment configuration
        :type yaml_data: dict
        """
        # TODO: Write record classes for such configurations to be transferred
        self.model = None
        self.yaml_data = yaml_data

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
        model_params = self.yaml_data["model_parameters"]
        model_path = model_params["general"]["model_path"]
        model_name = model_params["general"]["model_pseudoname"]
        library_type = model_params["general"]["library_type"]
        max_tokens = model_params["general"]["max_tokens"]
        ctx_width_tokens = model_params["general"]["ctx_width_tokens"]
        remove_str_list = model_params["general"]["disallowed_strings"]

        inference_threads = model_params["llama_cpp"]["inference_threads"]
        gpu_layers = model_params["llama_cpp"]["gpu_layers"]

        llm = None
        if library_type == "llama_cpp":
            llm = model.LlamaModel(
                model_path=model_path,
                name=model_name,
                max_out_tokens=max_tokens,
                seed=42,  # Random seed (this can be adjusted)
                remove_string_list=remove_str_list,
                ctx_width_tokens=ctx_width_tokens,
                inference_threads=inference_threads,
                gpu_layers=gpu_layers,
            )
        elif library_type == "transformers":
            llm = model.TransformersModel(
                model_path=model_path,
                name=model_name,
                max_out_tokens=max_tokens,
                remove_string_list=remove_str_list,
            )
        else:
            raise NotImplementedError(
                f"Unknown model type: {library_type}. Supported types: llama_cpp, transformers"
            )
        return llm
