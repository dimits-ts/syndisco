SPHINX_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$(dirname "$SPHINX_DIR")"
SOURCE_FILES_DIR="$ROOT_DIR/src"
HTML_OUT_DIR="$ROOT_DIR/docs"

# avoid recursively reading output as input on later invocations
rm -r $HTML_OUT_DIR
mkdir -p $HTML_OUT_DIR
touch "${HTML_OUT_DIR}/.nojekyll"

# create dirs if not exist
mkdir -p "$SPHINX_DIR/_static"
mkdir -p "$ROOT_DIR/docs"

sphinx-apidoc -o "$SPHINX_DIR/source" $SOURCE_FILES_DIR
sphinx-build -M html $SPHINX_DIR $HTML_OUT_DIR

# move the files where github pages can see them
mv "$HTML_OUT_DIR/html"/* $HTML_OUT_DIR