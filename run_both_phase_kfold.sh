#!/usr/bin/env bash
# Launches the both-phase 5-fold baseline-vs-augmented comparison used by
# both_phase_augmentation_kfold.ipynb. Folds are independent training runs
# with nothing to synchronize between them, so this runs each as a separate
# process, one per GPU, round-robin -- train.py's GPU selection is via
# CUDA_VISIBLE_DEVICES, fixed for the life of a process, so a single Python
# process (or notebook kernel) can never span more than one GPU; this is the
# only way to actually parallelize across GPUs. See the notebook's intro cell.
#
# Usage:
#   bash run_both_phase_kfold.sh          # uses every GPU nvidia-smi reports
#   bash run_both_phase_kfold.sh 0 2 3    # or pin it to specific GPU indices
#
# Auto-detaches from the terminal on launch (see below), so it survives a
# dropped SSH connection -- just run it directly, no need to prefix nohup/tmux
# yourself. Safe to re-run: any run whose run_config.json already has
# "finished" is skipped, same spirit as run_variant()'s already_done() in
# both_phase_augmentation.ipynb.

set -euo pipefail
cd "$(dirname "$0")"

# Detach from the controlling terminal so a dropped connection can't SIGHUP
# this or its children -- the same failure mode that cut a training run short
# in both_phase_augmentation.ipynb (kernel lost mid-run) can otherwise happen
# here too. KFOLD_DETACHED marks the re-exec'd instance so this doesn't loop;
# running under tmux/screen already protects you, but re-detaching is harmless.
if [ -z "${KFOLD_DETACHED:-}" ]; then
    LOG="$(pwd)/run_both_phase_kfold.out"
    echo "Detaching so this survives a dropped connection."
    echo "  progress (this script):  tail -f $LOG"
    echo "  progress (a given run):  tail -f logs_both_<run_name>.log"
    echo "  GPU usage:                nvidia-smi"
    KFOLD_DETACHED=1 setsid nohup bash "$(pwd)/$(basename "$0")" "$@" </dev/null >"$LOG" 2>&1 &
    disown
    exit 0
fi

K_FOLDS=5
LOSS=dice_ce
PHASE=both
EPOCHS=40
BATCH_SIZE=8
SPLIT_SEED=0   # FROZEN -- must match both_phase_augmentation_kfold.ipynb's SPLIT_SEED
# --seed is set to the fold index below (not a fixed constant): baseline and
# data_aug at the SAME fold still share a seed, so augment is the only thing
# that differs between them (the actual comparison) -- but different folds now
# get independently-initialized models instead of all 5 starting from bit-
# identical weights, which would otherwise make fold-to-fold variance reflect
# only the patient partition and not any real training-noise sampling.

if [ "$#" -gt 0 ]; then
    GPUS=("$@")
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
N_GPUS=${#GPUS[@]}
if [ "$N_GPUS" -eq 0 ]; then
    echo "No GPUs found (nvidia-smi returned nothing) -- pass GPU indices explicitly, e.g.:" >&2
    echo "  bash run_both_phase_kfold.sh 0 1" >&2
    exit 1
fi
echo "Using ${N_GPUS} GPU(s): ${GPUS[*]}"

already_finished() {
    local run_dir="checkpoints/${PHASE}_$1"
    [ -f "$run_dir/run_config.json" ] && grep -q '"finished"' "$run_dir/run_config.json"
}

job=0
pids=()
for fold in $(seq 0 $((K_FOLDS - 1))); do
    for arm in baseline data_aug; do
        run_name="${arm}_${LOSS}_kfold${fold}"

        if already_finished "$run_name"; then
            echo "skip  ${run_name} (already finished)"
            continue
        fi

        # Throttle to N_GPUS concurrent jobs -- GPUS/N_GPUS otherwise only
        # labelled which GPU a job's CUDA_VISIBLE_DEVICES gets via round-robin,
        # with nothing capping how many actually run at once, so e.g. a single
        # GPU would get every remaining job launched onto it simultaneously.
        # N_GPUS=1 (`bash run_both_phase_kfold.sh 0`) now means genuinely
        # sequential: one job runs to completion before the next starts.
        while [ "$(jobs -rp | wc -l)" -ge "$N_GPUS" ]; do
            wait -n
        done

        gpu=${GPUS[$((job % N_GPUS))]}
        job=$((job + 1))

        aug_flag=()
        [ "$arm" = "data_aug" ] && aug_flag=(--augment)

        echo "start ${run_name} on GPU ${gpu} (log: logs_${PHASE}_${run_name}.log)"
        CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            --phase "$PHASE" --loss "$LOSS" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
            --amp --k-folds "$K_FOLDS" --fold "$fold" \
            --split-seed "$SPLIT_SEED" --seed "$fold" \
            --select-on macro_dice --lr-schedule poly --no-test-eval \
            "${aug_flag[@]}" --run-name "$run_name" \
            > "logs_${PHASE}_${run_name}.log" 2>&1 &
        pids+=($!)

        # Stagger starts a little so, on a COLD cache, simultaneous first-time
        # DICOM/NIfTI parses don't race each other writing the same .npy file.
        # A no-op here for the both-phase cache, which is already fully built.
        sleep 5
    done
done

if [ "${#pids[@]}" -eq 0 ]; then
    echo "Nothing to run -- every fold/arm combination is already finished."
    exit 0
fi

echo "Launched ${#pids[@]} job(s); waiting..."
wait "${pids[@]}"
echo "All done. Check logs_${PHASE}_*.log for any that failed."
