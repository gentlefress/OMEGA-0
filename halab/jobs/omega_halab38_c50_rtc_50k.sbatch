#!/bin/bash -l
# ω-0 (HF keycharon/Omega-zero) fine-tuned on the 38-task HA-Lab data (datasets/ha-lab-dataset-38tasks), no GMR arms:
# DIM=76 -> 64 SONIC token + 12 fingers (2-a), DIM=78 -> + 2 camera (2-b). OMEGA-0's own training loop with the
# released checkpoint's recipe (halab/omega_halab_train.py): 50-row chunk, training-time RTC (delay 0..7),
# lr 1e-4 cosine, warmup 1000, global batch 256 = 8/GPU x 4 A40 x accumulation 8, 50k steps.
# One job = one 7-day long-qos leg (A40: ~15k steps per leg); checkpoints every 100 steps. If the
# leg ends before MAX_STEPS (time limit, node/NCCL failure), it resubmits itself from the newest complete
# checkpoint. Every 10k steps the model weights are hard-linked to <run>/keep-<step>/ (artifact.json +
# pytorch_model.bin, load with omega_halab_train.load_policy) so they survive checkpoint rotation.
# Submit from the OMEGA-0 root: code and .venv come from there ($SLURM_SUBMIT_DIR, or OMEGA_CODE); data, model
# assets and outputs from OMEGA_GB.
# First leg also builds the shared ω-0 dataset ($OM/halab38: hdf5 annotations + stats, same val split as GR00T).
#
#   cd <OMEGA-0> && DIM=76 sbatch halab/jobs/omega_halab38_c50_rtc_50k.sbatch
#   cd <OMEGA-0> && DIM=78 sbatch halab/jobs/omega_halab38_c50_rtc_50k.sbatch
#
# Written for Skynet (partition ha-lab, qos long, A40/L40S gres, --output/--error below): adapt the #SBATCH lines
# to another cluster. Paths: OMEGA_GB (datasets/ha-lab-dataset-38tasks/halab or OMEGA_HALAB_DATA, huggingface/,
# outputs/ or OMEGA_OUT_ROOT), OMEGA_ASSETS (Omega-zero/, external/{t5-base,Wan2.2_VAE.pth}, built data halab38/ or
# OMEGA_DATA), OMEGA_PY (default <OMEGA-0>/.venv). Setup and resume: halab/README.md.
#   cd <OMEGA-0> && OMEGA_GB=<base> DIM=76 sbatch halab/jobs/omega_halab38_c50_rtc_50k.sbatch

#SBATCH --job-name=omega38
#SBATCH --partition=ha-lab
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a40:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/omega38-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/omega38-%j.err

set -uo pipefail
CODE=${OMEGA_CODE:-$SLURM_SUBMIT_DIR}                 # OMEGA-0 checkout (halab/, .venv)
HALAB=$CODE/halab
if [ ! -f "$HALAB/omega_halab_train.py" ]; then
  echo "$HALAB/omega_halab_train.py missing: sbatch from the OMEGA-0 root (or set OMEGA_CODE)"; exit 1
fi
PY=${OMEGA_PY:-$CODE/.venv/bin/python}
GB=${OMEGA_GB:-/coc/flash2/jpark984/gr00t_baseline}  # datasets/, outputs/, huggingface/
OM=${OMEGA_ASSETS:-$GB/omega}                         # Omega-zero/, external/{t5-base,Wan2.2_VAE.pth}, halab38/
DIM=${DIM:-76}
MAX_STEPS=${MAX_STEPS:-50000}
KEEP_EVERY=${KEEP_EVERY:-10000}
EXP=${EXP:-halab38_omega0_${DIM}d_c50_rtc8_g256_50k}
OUT=${OMEGA_OUT_ROOT:-$GB/outputs}/$EXP
DATA=${OMEGA_DATA:-$OM/halab38}
DATA_PARENT=${OMEGA_HALAB_DATA:-$GB/datasets/ha-lab-dataset-38tasks/halab}  # LeRobot task folders
PORT=${PORT:-$((29710 + DIM - 76))}  # 29710 (76-D) / 29712 (78-D): unique per concurrent job
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-603000}  # 7 days - 30 min: room for the last save, final export and a resubmit
export HF_HOME=$GB/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$HALAB${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"
echo "$(date) $EXP on $(hostname), job $SLURM_JOB_ID"

if [ ! -f "$OM/external/Wan2.2_VAE.pth" ]; then
  echo "$OM/external/Wan2.2_VAE.pth missing (JEPA future-latent targets): download it first"; exit 1
fi
# Both runs share one dataset build; the second waits for the first.
flock "$DATA.lock" bash -c "[ -f '$DATA/stats.json' ] || '$PY' '$HALAB/omega_halab_data.py' \
  --data-parent '$DATA_PARENT' --out '$DATA'" || exit 1

for d in "$OUT"/checkpoint-*; do  # killed mid-save: no progress.json
  [ -d "$d" ] && [ ! -f "$d/progress.json" ] && case "${d##*/}" in *.incomplete-*) ;; *) mv "$d" "$d.incomplete-$SLURM_JOB_ID";; esac
done

keep() {
  local d name step k
  for d in "$OUT"/checkpoint-*; do
    name=${d##*/checkpoint-}
    case "$name" in ''|*[!0-9]*) continue;; esac
    [ -f "$d/progress.json" ] || continue
    step=$((10#$name))
    [ $((step % KEEP_EVERY)) -eq 0 ] || continue
    k=$OUT/keep-$step
    [ -f "$k/artifact.json" ] && continue
    mkdir -p "$k" && ln -f "$d/pytorch_model.bin" "$k/pytorch_model.bin" && cp "$d/progress.json" "$k/" \
      && cp "$OUT/artifact_template.json" "$k/artifact.json" && echo "$(date) kept $k"
  done
}
(while sleep 60; do keep; done) &
KEEPER=$!

timeout --signal=TERM --kill-after=120 "$TIMEOUT_SECONDS" srun --cpu-bind=none "$PY" -m torch.distributed.run \
  --nproc_per_node=4 --master_port "$PORT" "$HALAB/omega_halab_train.py" \
  --checkpoint "$OM/Omega-zero" --external-models "$OM/external" --data-root "$DATA" --output "$OUT" \
  --action-dim "$DIM" --chunk 50 --max-steps "$MAX_STEPS" --batch-size 8 --accumulation 8 --workers 6
STATUS=$?
kill "$KEEPER" 2>/dev/null; keep

latest() { ls -d "$OUT"/checkpoint-* 2>/dev/null | while read -r d; do n=${d##*/checkpoint-}
  case "$n" in ''|*[!0-9]*) ;; *) [ -f "$d/progress.json" ] && echo $((10#$n));; esac; done | sort -n | tail -n 1; }
STEP=$(latest); STEP=${STEP:-0}
echo "$(date) train exit $STATUS, newest complete checkpoint: step $STEP"
if [ "$STEP" -ge "$MAX_STEPS" ] || [ -d "$OUT/policy-$(printf %08d "$MAX_STEPS")" ]; then
  echo "$EXP: done"; exit 0
fi
PREV=${PREV_STEP:--1}
FAILS=${NO_PROGRESS:-0}
# A leg can die before training (e.g. NCCL hang at DDP init): retry, stop after 3 such legs in a row.
if [ "$STEP" -le "$PREV" ]; then FAILS=$((FAILS + 1)); else FAILS=0; fi
if [ "$FAILS" -ge 3 ]; then
  echo "$EXP: $FAILS legs in a row without progress (step $STEP): chain stopped" >&2; exit 1
fi
# Slurm once lost an accepted submission (its job ID went to another user's job): confirm the next leg is ours.
for try in 1 2 3; do
  NEXT=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
    --export="ALL,DIM=$DIM,EXP=$EXP,PREV_STEP=$STEP,NO_PROGRESS=$FAILS" \
    "${SUBMIT_SCRIPT:-$HALAB/jobs/omega_halab38_c50_rtc_50k.sbatch}") || { sleep 30; continue; }
  NEXT=${NEXT%%;*}
  sleep 120
  if [ "$(squeue -h -j "$NEXT" -o %u 2>/dev/null)" = "$USER" ]; then echo "$(date) next leg queued: $NEXT"; exit 0; fi
  echo "$(date) next leg $NEXT is not in the queue as $USER's job (try $try): resubmitting" >&2
done
echo "$EXP: could not queue the next leg: chain stopped" >&2; exit 1
