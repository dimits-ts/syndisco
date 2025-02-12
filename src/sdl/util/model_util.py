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
            # dynamically load library to avoid dependency hell
            from sdl.backend.cpp_model import LlamaModel

            llm = LlamaModel(
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
            # dynamically load library to avoid dependency hell
            from sdl.backend.trans_model import TransformersModel

            llm = TransformersModel(
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
