import llama_cpp
import typing

from . import model


class LlamaModel(model.Model):

    def __init__(
        self,
        model_path: str,
        name: str,
        gpu_layers: int,
        seed: int = 42,
        ctx_width_tokens: int = 2048,
        max_out_tokens: int = 400,
        inference_threads: int = 3,
        remove_string_list: list[str] = [],
    ):
        """
        Initialize a new LLM wrapper.

        :param model_path: the LLM to be used
        :type model_path: llama_cpp.Llama
        :param name: a shorthand name for the model used
        :type name: str
        :param max_out_tokens: the maximum number of tokens in the response
        :type max_out_tokens: int
        :param seed: random seed
        :type seed: int
        :param ctx_width_tokens: the number of tokens available for context
        :type ctx_width_tokens: int
        :param inference_threads: how many CPU threads will run on the RAM-allocated tensors
        :type inference_threads: int
        :param gpu_layers: how many layers will be offloaded to the GPU
        :type gpu_layers: int
        :param remove_string_list: a list of strings to be removed from the response.
        Used to prevent model-specific conversational collapse, defaults to []
        :type remove_string_list: list, optional
        """
        super().__init__(remove_string_list)

        self.model = llama_cpp.Llama(
            model_path=model_path,
            seed=seed,
            n_ctx=ctx_width_tokens,
            n_threads=inference_threads,
            n_gpu_layers=gpu_layers,
            use_mmap=True,
            chat_format="alpaca",
            mlock=True,
            verbose=False,
        )
        self.max_out_tokens = max_out_tokens
        self.seed = seed
        self.name = name

    def generate_response(
        self, json_prompt: tuple[typing.Any, typing.Any], stop_words: list[str]
    ) -> str:
        output = self.model.create_chat_completion(
            messages=json_prompt,  # type: ignore
            max_tokens=self.max_out_tokens,
            seed=self.seed,
            stop=stop_words,
        )  # prevent model from generating the next actor's response

        response = self._get_response_from_output(output)

        return response

    @staticmethod
    def _get_response_from_output(json_output) -> str:
        """
        Extracts the model's response from the raw output as a string.
        """
        return json_output["choices"][0]["message"]["content"]
