cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda activate movens-interact
python -m simulation.prepare_assets "$@"
