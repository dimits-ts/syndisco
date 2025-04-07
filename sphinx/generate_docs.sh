SPHINX_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$(dirname "$SPHINX_DIR")"
SOURCE_FILES_DIR="$ROOT_DIR/src"
HTML_OUT_DIR="$ROOT_DIR/docs"

# avoid recursively reading output as input on later invocations
rm -rf "$SPHINX_DIR/doctrees"
rm -rf "$SPHINX_DIR/html"

# create dirs if not exist
mkdir -p "$SPHINX_DIR/_static"
mkdir -p "$ROOT_DIR/docs"

sphinx-apidoc -o "$SPHINX_DIR/source" $SOURCE_FILES_DIR
sphinx-build -M html $SPHINX_DIR $HTML_OUT_DIR

# move the files where github pages can see them
mv "$HTML_OUT_DIR/html"/* $HTML_OUT_DIR