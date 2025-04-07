SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SOURCE_FILES_DIR="$ROOT_DIR/src"

rm -rf "$SCRIPT_DIR/doctrees"
rm -rf "$SCRIPT_DIR/html"
mkdir -p "$SCRIPT_DIR/_static"
sphinx-apidoc -o "$SCRIPT_DIR/source" $SOURCE_FILES_DIR
sphinx-build -M html $SCRIPT_DIR $SCRIPT_DIR