if [ -n "$ZSH_VERSION" ]; then
    THIS_DIR=$(dirname $(readlink -f "${(%):-%N}"))
elif [ -n "$BASH_VERSION" ]; then
    THIS_DIR=$(dirname $(readlink -f "${BASH_SOURCE[0]}"))
fi

ROOT_DIR=$(cd "$THIS_DIR/.." && pwd)

export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
