import llama_cpp
import argparse

from sdl.backend import models
from sdl.serialization import conversation_io
from sdl.util import file_util

REMOVE_STR_LIST = ["```"]

def main():
    parser = argparse.ArgumentParser(description="Run synthetic dialogue via Llama conversation model.")
    parser.add_argument('--input_file', required=True, help="Input conversation file path.")
    parser.add_argument('--output_dir', required=True, help="Output directory path.")
    parser.add_argument('--model_path', required=True, help="Model file path.")
    parser.add_argument('--model_name', required=True, help="Name of the model.")
    parser.add_argument('--type', required=True, choices=['llama', 'transformers'], help="Type of model to use.")
    parser.add_argument('--max_tokens', type=int, default=512, help="Maximum number of tokens.")
    parser.add_argument('--ctx_width_tokens', type=int, default=1024, help="Context width in tokens.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--inference_threads', type=int, default=4, help="Number of threads for inference.")
    parser.add_argument('--gpu_layers', type=int, default=12, help="Number of layers offloaded to the GPU (requires CUDA).")

    args = parser.parse_args()

    input_file_path = args.input_file
    output_dir = args.output_dir
    max_tokens = args.max_tokens
    ctx_width_tokens = args.ctx_width_tokens
    model_path = args.model_path
    model_name = args.model_name
    model_type = args.type
    random_seed = args.random_seed
    inference_threads = args.inference_threads
    gpu_layers = args.gpu_layers

    print("Loading LLM...")

    # Switch-like dictionary for model selection
    model_switch = {
        "llama": lambda: models.LlamaModel(
            llama_cpp.Llama(
                model_path=model_path,
                seed=random_seed,
                n_ctx=ctx_width_tokens,
                n_threads=inference_threads,
                n_gpu_layers=gpu_layers,
                use_mmap=True,
                chat_format="alpaca",
                mlock=True,
                verbose=False,
            ),
            name=model_name,
            max_out_tokens=max_tokens,
            seed=random_seed,
            remove_string_list=REMOVE_STR_LIST
        ),
        "transformers": lambda: models.TransformersModel(
            model_path=model_path,
            name=model_name,
            max_out_tokens=max_tokens,
            remove_string_list=REMOVE_STR_LIST
        )
    }

    if model_type not in model_switch:
        raise NotImplementedError(f"Unknown model type: {model_type}. Supported types: llama, transformers")

    # Load the selected model
    model = model_switch[model_type]()
    print("Model loaded.")

    # Load data and start conversation
    data = conversation_io.LLMConvData.from_json_file(input_file_path)
    generator = conversation_io.LLMConvGenerator(data=data, user_model=model, moderator_model=model)
    conv = generator.produce_conversation()

    print("Beginning conversation...")
    conv.begin_conversation(verbose=True)
    output_path = file_util.generate_datetime_filename(output_dir=output_dir, file_ending=".json")
    conv.to_json_file(output_path)
    print("Conversation saved to ", output_path)


if __name__ == "__main__":
    main()
