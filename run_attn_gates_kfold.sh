#!/usr/bin/env bash
# Launches the 5-fold attention-GATE runs (--arch unet_attn_gates). Sibling of
# run_context_slices_kfold.sh and run_both_phase_kfold.sh -- here the arm axis
# is the architecture. The control arm it is meant to be compared against is
# both_data_aug_dice_ce_kfold{0..4} (already finished: phase=both,
# augment=True, loss=dice_ce, context_slices=0, split_seed=0, seed=fold,
# 40 epochs, batch 8, scale 0.5, poly LR, select-on macro_dice -- exactly this
# script's config apart from --arch), so there is nothing to train for it.
#
# THIS is the clean attention ablation: unet/attention_gate_unet.py is the
# milesial UNet with MONAI's AttentionBlock on each skip and nothing else
# changed (verified: every UNet parameter name and shape is preserved, the
# gates add 84 tensors / +1.13% params). A difference against the control is
# attributable to the gates alone. Contrast run_attention_kfold.sh, which
# trains MONAI's whole AttentionUnet network and is NOT attention-only.
#
# The learning rate is left at train.py's 1e-5 default, the same value every
# existing run used. It was never tuned per-architecture, and re-tuning it for
# only this arm would confound the comparison; a fair "with tuning" claim needs
# both arms swept, which this script does not do.
#
# Folds are independent training runs with nothing to synchronize between
# them, so this runs each as a separate process, one per GPU, round-robin --
# train.py's GPU selection is via CUDA_VISIBLE_DEVICES, fixed for the life of
# a process, so a single Python process (or notebook kernel) can never span
# more than one GPU; this is the only way to actually parallelize across GPUs.
# Pass a single GPU index to force strictly sequential execution.
#
# Usage:
#   bash run_attn_gates_kfold.sh          # uses every GPU nvidia-smi reports
#   bash run_attn_gates_kfold.sh 0 2 3    # or pin it to specific GPU indices
#
# Auto-detaches from the terminal on launch (see below), so it survives a
# dropped SSH connection -- just run it directly, no need to prefix nohup/tmux
# yourself. Safe to re-run: any run whose run_config.json already has
# "finished" is skipped.

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${KFOLD_DETACHED:-}" ]; then
    LOG="$(pwd)/run_attn_gates_kfold.out"
    echo "Detaching so this survives a dropped connection."
    echo "  progress (this script):  tail -f $LOG"
    echo "  progress (a given run):  tail -f logs_both_attngate_kfold*.log"
    echo "  GPU usage:                nvidia-smi"
    KFOLD_DETACHED=1 setsid nohup bash "$(pwd)/$(basename "$0")" "$@" </dev/null >"$LOG" 2>&1 &
    disown
    exit 0
fi

K_FOLDS=5
PHASE=both
LOSS=dice_ce
ARCH=unet_attn_gates
EPOCHS=40
BATCH_SIZE=8
SPLIT_SEED=0             # FROZEN -- must match every other both-phase k-fold run
RUN_PREFIX=attngate

# Matches the both_data_aug_dice_ce_kfold* control arm: every arm in this
# comparison trains with augmentation on, so the comparison is architecture vs
# architecture, not confounded with augmentation on/off.
AUGMENT=true

# Plain `python`/`python3` on this node resolves to the system FSL install,
# which has no torch -- a plain `bash script.sh` invocation gets neither a
# notebook kernel nor an interactive shell's PATH, so it has to be pointed at
# the venv explicitly.
PYTHON=/hpc/jgeo610/Virtual_ENV/unet_env/bin/python
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    echo "FATAL: $PYTHON can't import torch -- check the venv still exists at that path." >&2
    exit 1
fi
# monai is this arm's extra dependency (only --arch attention_unet needs it),
# so fail here rather than 5 times over inside train.py.
if ! "$PYTHON" -c "import monai" 2>/dev/null; then
    echo "FATAL: $PYTHON can't import monai -- install it with:" >&2
    echo "  $PYTHON -m pip install --no-deps 'monai==1.6.0'" >&2
    echo "  (--no-deps is REQUIRED: this venv's torch is a +cu118 build that pip" >&2
    echo "   does not see as installed, and a plain install replaces it.)" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    GPUS=("$@")
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
N_GPUS=${#GPUS[@]}
if [ "$N_GPUS" -eq 0 ]; then
    echo "No GPUs found (nvidia-smi returned nothing) -- pass GPU indices explicitly, e.g.:" >&2
    echo "  bash run_attn_gates_kfold.sh 0 1" >&2
    exit 1
fi
echo "Using ${N_GPUS} GPU(s): ${GPUS[*]}"

already_finished() {
    local run_dir="checkpoints/${PHASE}_$1"
    [ -f "$run_dir/run_config.json" ] && grep -q '"finished"' "$run_dir/run_config.json"
}

job=0
pids=()
# --seed is the fold index, matching both_data_aug_dice_ce_kfold*'s convention
# (confirmed in their run_config.json: seed == fold), so at a given fold the
# only thing differing between control and this arm is --arch.
for fold in $(seq 0 $((K_FOLDS - 1))); do
    run_name="${RUN_PREFIX}_kfold${fold}"

    if already_finished "$run_name"; then
        echo "skip  ${run_name} (already finished)"
        continue
    fi

    # Throttle to N_GPUS concurrent jobs.
    while [ "$(jobs -rp | wc -l)" -ge "$N_GPUS" ]; do
        wait -n
    done

    gpu=${GPUS[$((job % N_GPUS))]}
    job=$((job + 1))

    aug_flag=()
    [ "$AUGMENT" = true ] && aug_flag=(--augment)

    echo "start ${run_name} on GPU ${gpu} (log: logs_${PHASE}_${run_name}.log)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" train.py \
        --arch "$ARCH" \
        --phase "$PHASE" --loss "$LOSS" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
        --amp --k-folds "$K_FOLDS" --fold "$fold" \
        --split-seed "$SPLIT_SEED" --seed "$fold" \
        --select-on macro_dice --lr-schedule poly --no-test-eval \
        "${aug_flag[@]}" --run-name "$run_name" \
        > "logs_${PHASE}_${run_name}.log" 2>&1 &
    pids+=($!)

    # Stagger starts a little so, on a COLD cache, simultaneous first-time
    # DICOM/NIfTI parses don't race each other writing the same .npy file. A
    # no-op here for the both-phase cache, which is already fully built.
    sleep 5
done

if [ "${#pids[@]}" -eq 0 ]; then
    echo "Nothing to run -- every fold is already finished."
    exit 0
fi

echo "Launched ${#pids[@]} job(s); waiting..."
wait "${pids[@]}"
echo "All done. Check logs_${PHASE}_attngate_kfold*.log for any that failed."
