cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda activate movens-interact
python -B scripts/download_models.py "$@"
