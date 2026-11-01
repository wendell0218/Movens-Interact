cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda activate movens-interact
export MUJOCO_GL="egl"
python -m simulation.render "$@"
