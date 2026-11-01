cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda activate movens-interact
python -B -u training/collect.py "$@"
