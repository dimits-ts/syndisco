import llama_cpp
import lib.conversation
import lib.models
import lib.util
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run synthetic dialogue via Llama conversation model.")
    parser.add_argument('--input_file', required=True, help="Input conversation file path.")
    parser.add_argument('--output_dir', required=True, help="Output directory path.")
    parser.add_argument('--model_path', required=True, help="Model file path.")
    parser.add_argument('--max_tokens', type=int, default=512, help="Maximum number of tokens.")
    parser.add_argument('--ctx_width_tokens', type=int, default=1024, help="Context width in tokens.")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--inference_threads', type=int, default=4, help="Number of threads for inference.")
    parser.add_argument('--gpu_layers', type=int, default=12, help="Number of layers offloaded to the GPU (requires "
                                                                   "CUDA).")

    args = parser.parse_args()

    input_file_path = args.input_file
    output_dir = args.output_dir
    max_tokens = args.max_tokens
    ctx_width_tokens = args.ctx_width_tokens
    model_path = args.model_path
    random_seed = args.random_seed
    inference_threads = args.inference_threads
    gpu_layers = args.gpu_layers

    print("Loading LLM...")
    llm = llama_cpp.Llama(
        model_path=model_path,
        seed=random_seed,
        n_ctx=ctx_width_tokens,
        n_threads=inference_threads,
        n_gpu_layers=gpu_layers,  # will vary from machine to machine
        use_mmap=True,  # if ran on Linux, model size does not matter since the model uses mmap for lazy loading
        chat_format="alpaca",  # using llama-2 leads to well-known model collapse
        mlock=True,  # keep memcached model files in RAM if possible
        verbose=False,
    )
    print("Model loaded.")

    model = lib.models.LlamaModel(llm, max_out_tokens=max_tokens, seed=random_seed)
    data = lib.conversation.LLMConvData.from_json_file(input_file_path)
    generator = lib.conversation.LLMConvGenerator(data=data, user_model=model, moderator_model=model)
    conv = generator.produce_conversation()

    print("Beginning conversation...")
    conv.begin_conversation(verbose=True)
    output_path = lib.util.generate_datetime_filename(output_dir=output_dir, file_ending=".json")
    conv.to_json_file(output_path)
    print("Conversation saved to ", output_path)


if __name__ == "__main__":
    main()
