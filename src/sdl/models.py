import llama_cpp


class LlamaModel:

    @staticmethod
    def _get_response_from_output(json_output) -> str:
        """
        Extracts the model's response from the raw output as a string.
        """
        return json_output["choices"][0]["message"]["content"]
    

    def __init__(self, model: llama_cpp.Llama, max_out_tokens: int, seed: int):
        self.model = model
        self.max_out_tokens = max_out_tokens
        self.seed = seed

    def prompt(self, json_prompt: list[llama_cpp.ChatCompletionRequestMessage], stop_list: list[str]) -> str:
        output = self.model.create_chat_completion(
                        messages=json_prompt,
                        max_tokens=self.max_out_tokens,
                        seed=self.seed,
                        stop=["###", "\n\n"] + stop_list) # prevent model from generating the next actor's response
        
        response = self._get_response_from_output(output)

        return response
 

