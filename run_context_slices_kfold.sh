#!/usr/bin/env bash
# Launches the 5-fold 2.5D (context_slices=1) runs used by
# context_slices_kfold_diagnostics.ipynb. Sibling of run_both_phase_kfold.sh
# and run_both_phase_loss_ablation_kfold.sh -- here the arm axis is
# context_slices instead of augment on/off or loss. Only ONE arm's worth of
# runs is launched: the 2D arm the notebook compares against reuses
# both_data_aug_dice_ce_kfold{0..4} (already finished, phase=both,
# augment=True, loss=dice_ce, split_seed=0, seed=fold -- exactly this script's
# config apart from context_slices), so there is nothing to train for it.
#
# Folds are independent training runs with nothing to synchronize between
# them, so this runs each as a separate process, one per GPU, round-robin --
# train.py's GPU selection is via CUDA_VISIBLE_DEVICES, fixed for the life of
# a process, so a single Python process (or notebook kernel) can never span
# more than one GPU; this is the only way to actually parallelize across
# GPUs. See the notebook's intro cell.
#
# Usage:
#   bash run_context_slices_kfold.sh          # uses every GPU nvidia-smi reports
#   bash run_context_slices_kfold.sh 0 2 3    # or pin it to specific GPU indices
#
# Auto-detaches from the terminal on launch (see below), so it survives a
# dropped SSH connection -- just run it directly, no need to prefix nohup/tmux
# yourself. Safe to re-run: any run whose run_config.json already has
# "finished" is skipped.

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${KFOLD_DETACHED:-}" ]; then
    LOG="$(pwd)/run_context_slices_kfold.out"
    echo "Detaching so this survives a dropped connection."
    echo "  progress (this script):  tail -f $LOG"
    echo "  progress (a given run):  tail -f logs_both_ctx_2p5d_kfold*.log"
    echo "  GPU usage:                nvidia-smi"
    KFOLD_DETACHED=1 setsid nohup bash "$(pwd)/$(basename "$0")" "$@" </dev/null >"$LOG" 2>&1 &
    disown
    exit 0
fi

K_FOLDS=5
PHASE=both
LOSS=dice_ce
CONTEXT_SLICES=1        # the one new axis this script trains -- see the notebook's
                         # CONTEXT_SLICES_FOR['2p5d']; must match it
EPOCHS=40
BATCH_SIZE=8
SPLIT_SEED=0             # FROZEN -- must match context_slices_kfold_diagnostics.ipynb's SPLIT_SEED
RUN_PREFIX=ctx
RUN_NAME_TEMPLATE="${RUN_PREFIX}_2p5d_kfold"   # must match the notebook's run_name_for('2p5d', fold)

# Matches both_phase_augmentation_kfold_diagnostics.ipynb's data_aug arm and
# both_phase_loss_ablation_kfold_diagnostics.ipynb: every arm in this
# comparison trains with augmentation on, so the comparison is architecture
# vs architecture, not confounded with augmentation on/off.
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

if [ "$#" -gt 0 ]; then
    GPUS=("$@")
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
N_GPUS=${#GPUS[@]}
if [ "$N_GPUS" -eq 0 ]; then
    echo "No GPUs found (nvidia-smi returned nothing) -- pass GPU indices explicitly, e.g.:" >&2
    echo "  bash run_context_slices_kfold.sh 0 1" >&2
    exit 1
fi
echo "Using ${N_GPUS} GPU(s): ${GPUS[*]}"

already_finished() {
    local run_dir="checkpoints/${PHASE}_$1"
    [ -f "$run_dir/run_config.json" ] && grep -q '"finished"' "$run_dir/run_config.json"
}

job=0
pids=()
# --seed is set to the fold index (not a fixed constant), matching
# both_data_aug_dice_ce_kfold*'s convention (confirmed in their
# run_config.json: seed == fold): every fold gets an independently-initialized
# model/augmentation stream, and the SAME seed convention across the 2d and
# 2p5d arms means, at a given fold, only context_slices differs between them.
for fold in $(seq 0 $((K_FOLDS - 1))); do
    run_name="${RUN_NAME_TEMPLATE}${fold}"

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
        --phase "$PHASE" --loss "$LOSS" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
        --context-slices "$CONTEXT_SLICES" \
        --amp --k-folds "$K_FOLDS" --fold "$fold" \
        --split-seed "$SPLIT_SEED" --seed "$fold" \
        --select-on macro_dice --lr-schedule poly --no-test-eval \
        "${aug_flag[@]}" --run-name "$run_name" \
        > "logs_${PHASE}_${run_name}.log" 2>&1 &
    pids+=($!)

    # Stagger starts a little so, on a COLD cache, simultaneous first-time
    # DICOM/NIfTI parses don't race each other writing the same .npy file. A
    # no-op here for the both-phase cache, which is already fully built by
    # earlier both_* notebooks.
    sleep 5
done

if [ "${#pids[@]}" -eq 0 ]; then
    echo "Nothing to run -- every fold is already finished."
    exit 0
fi

echo "Launched ${#pids[@]} job(s); waiting..."
wait "${pids[@]}"
echo "All done. Check logs_${PHASE}_${RUN_NAME_TEMPLATE}*.log for any that failed."
