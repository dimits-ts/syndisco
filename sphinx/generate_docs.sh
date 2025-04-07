SPHINX_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$(dirname "$SPHINX_DIR")"
SOURCE_FILES_DIR="$ROOT_DIR/src"
HTML_OUT_DIR="$ROOT_DIR/docs"

rm -rf "$SPHINX_DIR/doctrees"
rm -rf "$SPHINX_DIR/html"
mkdir -p "$SPHINX_DIR/_static"
mkdir -p "$ROOT_DIR/docs"
sphinx-apidoc -o "$SPHINX_DIR/source" $SOURCE_FILES_DIR
sphinx-build -M html $SPHINX_DIR $HTML_OUT_DIR