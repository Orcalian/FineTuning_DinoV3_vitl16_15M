#!/bin/bash
#SBATCH --job-name=dinov3_train
#SBATCH --output=/mnt/beegfs02/scratch/a_claveau/slurm_logs/dinov3_train_%j.out
#SBATCH --error=/mnt/beegfs02/scratch/a_claveau/slurm_logs/dinov3_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=gpgpuq

# =============================================================================
# Entrainement DINOv3 vit_large UNIQUE, parametre par le SEUL fichier de config.
#
#   sbatch [--reservation=NOM] ~/lancer_dinov3_train.sh CONFIG.yaml [OUT_DIR]
#
# Exemples :
#   sbatch ~/lancer_dinov3_train.sh ~/dinov3/dinov3/configs/train/dinov3_vitl16_380k.yaml
#   sbatch ~/lancer_dinov3_train.sh ~/dinov3/dinov3/configs/train/dinov3_vitl16_15M.yaml
#
# - OUT_DIR est DERIVE DU NOM de la config si non fourni :
#     dinov3_vitl16_380k.yaml -> /mnt/beegfs02/scratch/a_claveau/dinov3_vitl16_380k_outputs
#   (2e argument optionnel pour le forcer)
# - MAX_ITER est LU DEPUIS LA CONFIG (epochs x OFFICIAL_EPOCH_LENGTH) :
#   la duree du run est donc entierement pilotee par le yaml.
# - Chaine de segments anti-fuite : chaque job fait ~ITERS_PER_SEGMENT
#   iterations puis s'arrete apres un checkpoint et se resoumet lui-meme
#   (reprise DCP automatique depuis OUT_DIR/ckpt), jusqu'a MAX_ITER.
# - La reservation du job courant est propagee automatiquement aux
#   resoumissions (via SLURM_JOB_RESERVATION).
# =============================================================================

set -uo pipefail

# ---- arguments ----
CONFIG="${1:?usage: sbatch lancer_dinov3_train.sh CONFIG.yaml [OUT_DIR]}"
SCRATCH="/mnt/beegfs02/scratch/a_claveau"
CFG_NAME=$(basename "$CONFIG" .yaml)
OUT_DIR="${2:-$SCRATCH/${CFG_NAME}_outputs}"     # derive du nom de la config
SCRIPT_PATH="$HOME/lancer_dinov3_train.sh"       # chemin reel pour la resoumission

[ -f "$CONFIG" ] || { echo "ERREUR: config introuvable: $CONFIG"; exit 1; }
mkdir -p "$OUT_DIR" "$SCRATCH/slurm_logs"

# ---- MAX_ITER lu depuis la config (epochs x OFFICIAL_EPOCH_LENGTH) ----
EPOCHS=$(grep -E '^  epochs:' "$CONFIG" | head -1 | awk '{print $2}')
EPLEN=$(grep -E '^  OFFICIAL_EPOCH_LENGTH:' "$CONFIG" | head -1 | awk '{print $2}')
if [ -z "$EPOCHS" ] || [ -z "$EPLEN" ]; then
    echo "ERREUR: impossible de lire epochs/OFFICIAL_EPOCH_LENGTH dans $CONFIG"
    exit 1
fi
MAX_ITER=$(( EPOCHS * EPLEN ))

# Combien d'iterations par segment avant arret propre (apres un checkpoint)
ITERS_PER_SEGMENT=8000

# ==========================
# Chemins / environnement
# ==========================
export REPO_DIR="/home/a_claveau/dinov3"

module load miniconda/25.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dinov3_py311

cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-48} / 4 ))
export WANDB_MODE=disabled
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MALLOC_TRIM_THRESHOLD_=0

# ==========================
# Logs environnement
# ==========================
echo "=========================================="
echo "Job ID : ${SLURM_JOB_ID:-unknown}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "Date   : $(date)"
echo "REPO_DIR = $REPO_DIR"
echo "CONFIG   = $CONFIG"
echo "OUT_DIR  = $OUT_DIR"
echo "MAX_ITER = $MAX_ITER  (= $EPOCHS epochs x $EPLEN)"
echo "RESA     = ${SLURM_JOB_RESERVATION:-aucune}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

python - <<'PY'
import torch
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
n = torch.cuda.device_count()
print("GPU count:", n)
for i in range(n):
    print(f"  GPU {i}:", torch.cuda.get_device_name(i),
          "capability", torch.cuda.get_device_capability(i))
print("sm_90 dans arch_list:", "sm_90" in torch.cuda.get_arch_list())
assert torch.cuda.is_available(), "CUDA indisponible."
assert n == 4, f"4 GPU attendus, {n} visibles."
PY

echo "=========================================="
echo "Lancement entrainement DINOv3 vit_large (torchrun, 4 GPU)..."
echo "=========================================="

# (l'init depuis les poids LVD-1689M passe par resume_from_teacher_chkpt dans le YAML)

# iteration de depart de CE segment = dernier checkpoint present
START_CKPT=$(ls "$OUT_DIR/ckpt/" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
START_CKPT=${START_CKPT:-0}
STOP_AT=$(( START_CKPT + ITERS_PER_SEGMENT ))
echo "Segment : depart=$START_CKPT, arret vise a $STOP_AT (max $MAX_ITER)"

# ---- lancer torchrun en arriere-plan ----
torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    dinov3/train/train.py \
    --config-file "$CONFIG" \
    --output-dir "$OUT_DIR" &
TORCHRUN_PID=$!

# ---- watcher : arrete proprement le torchrun apres un checkpoint >= STOP_AT ----
(
    while kill -0 "$TORCHRUN_PID" 2>/dev/null; do
        sleep 60
        LAST=$(ls "$OUT_DIR/ckpt/" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
        LAST=${LAST:-0}
        if [ "$LAST" -ge "$STOP_AT" ] || [ "$LAST" -ge "$(( MAX_ITER - 1 ))" ]; then
            echo "[watcher] checkpoint $LAST atteint (cible $STOP_AT) -> arret propre du torchrun"
            kill -TERM "$TORCHRUN_PID" 2>/dev/null || true
            sleep 30
            kill -KILL "$TORCHRUN_PID" 2>/dev/null || true
            break
        fi
    done
) &
WATCHER_PID=$!

# attendre la fin du torchrun sans faire planter le script
set +e
wait "$TORCHRUN_PID"
set -e
kill "$WATCHER_PID" 2>/dev/null || true

# ---- resoumission tant que pas termine ----
LAST_CKPT=$(ls "$OUT_DIR/ckpt/" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
LAST_CKPT=${LAST_CKPT:-0}
echo "Fin de segment. Dernier checkpoint : $LAST_CKPT / $MAX_ITER"
if [ "$LAST_CKPT" -lt "$(( MAX_ITER - 1 ))" ]; then
    echo "Incomplet -> resoumission d'un nouveau segment (memes CONFIG/OUT_DIR)"
    RESA_OPT=""
    [ -n "${SLURM_JOB_RESERVATION:-}" ] && RESA_OPT="--reservation=$SLURM_JOB_RESERVATION"
    sbatch $RESA_OPT "$SCRIPT_PATH" "$CONFIG" "$OUT_DIR"
else
    echo "Entrainement TERMINE ($LAST_CKPT >= $MAX_ITER)"
fi
echo "=========================================="
echo "Fin du job : $(date)"
echo "=========================================="
