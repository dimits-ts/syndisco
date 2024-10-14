# Function to check if a file exists and download if it doesn't
function download_if_not_exists() {
    local file_path=$1
    local download_url=$2

    # Check if the file exists
    if [ ! -f "$file_path" ]; then
        echo "File $file_path does not exist. Downloading..."
        download_from_huggingface "$file_path" "$download_url"
    else
        echo "File $file_path already exists. Skipping download."
    fi
}

# Function to handle downloading from Huggingface
function download_from_huggingface() {
    local file_path=$1
    local download_url=$2

    wget "$download_url" -O "$file_path"
}

download_if_not_exists "../models/llama-3-8B.gguf" "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B.Q4_K_M.gguf?download=true"
download_if_not_exists "../models/llama-3-8B-instruct.gguf" "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf?download=true"
