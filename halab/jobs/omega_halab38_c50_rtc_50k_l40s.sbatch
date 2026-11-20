#!/bin/bash -l
# omega_halab38_c50_rtc_50k.sbatch on 4x L40S instead of 4x A40: same recipe, its own run
# (outputs/halab38_omega0_<DIM>d_c50_rtc8_g256_50k_l40s, own wandb run, port 29714/29716), so it runs alongside the A40 run.
# A resubmitted leg uses this file, so it stays on L40S. Same OMEGA_* overrides as the A40 file.
#
#   cd <OMEGA-0> && DIM=76 sbatch halab/jobs/omega_halab38_c50_rtc_50k_l40s.sbatch

#SBATCH --job-name=omega38l
#SBATCH --partition=ha-lab
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
# bishop: 5 legs (10-29, 11-01) died within 15 min of start, before any step (NCCL errors at DDP init).
#SBATCH --exclude=bishop
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --output=/coc/flash2/jpark984/gr00t_baseline/omega38l-%j.out
#SBATCH --error=/coc/flash2/jpark984/gr00t_baseline/omega38l-%j.err

JOBS=${OMEGA_CODE:-$SLURM_SUBMIT_DIR}/halab/jobs  # submit from the OMEGA-0 root
export SUBMIT_SCRIPT=$JOBS/omega_halab38_c50_rtc_50k_l40s.sbatch
export EXP=${EXP:-halab38_omega0_${DIM:-76}d_c50_rtc8_g256_50k_l40s}
export PORT=${PORT:-$((29714 + ${DIM:-76} - 76))}
source "$JOBS/omega_halab38_c50_rtc_50k.sbatch"
