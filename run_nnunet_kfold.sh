#!/usr/bin/env bash
# Trains nnU-Net v2 (3d_fullres, 250-epoch trainer) on the 5 folds of the
# cardiac dataset exported by export_nnunet_dataset.py.
#
# Unlike every other run_*_kfold.sh in this repo, this one does NOT call
# train.py. nnU-Net is a framework, not a model: it owns preprocessing,
# resampling, patch-size and architecture planning, training and inference, and
# its authors' claim is that those are what win rather than the network. So it
# cannot be an --arch; it runs as its own pipeline, in its own venv
# (/hpc/jgeo610/Virtual_ENV/nnunet_env -- unet_env is never touched), and is
# bridged back into this repo at the metrics layer by
# score_nnunet_predictions.py.
#
# The control arm is the already-finished both_data_aug_dice_ce_kfold{0..4}.
# Fold membership is identical by construction: make_nnunet_splits.py copies
# it verbatim out of those runs' run_config.json into splits_final.json, which
# is what lets metrics.compare_runs pair the two arms patient-by-patient.
#
# NOTE this is 3d_fullres against a 2D control, so part of any difference is
# 3D-vs-2D rather than nnU-Net-vs-UNet. That is a reporting caveat, not
# something this script can resolve.
#
# No --npz: that dumps full softmax volumes (6 classes x 110x432x432 per case)
# and is only needed for cross-configuration ensembling, which this arm does
# not do. nnU-Net still writes the validation segmentations that
# score_nnunet_predictions.py consumes, into fold_<F>/validation/.
#
# Usage:
#   bash run_nnunet_kfold.sh              # auto-picks GPUs that are actually idle
#   bash run_nnunet_kfold.sh 2 3 8        # or pin it to specific GPU indices
#   DRY_RUN=1 bash run_nnunet_kfold.sh    # show the plan, launch nothing
#
# Knobs (all environment variables, all with sane defaults):
#   N_PROC_DA=6        data-augmentation worker processes PER FOLD
#   MAX_CONCURRENT=5   folds training simultaneously
#   MIN_FREE_MIB=16000 skip GPUs with less free memory than this
#   MAX_UTIL_PCT=25    skip GPUs busier than this
#
# e.g. to tread lightly on a busy node:
#   N_PROC_DA=3 MAX_CONCURRENT=2 bash run_nnunet_kfold.sh
#
# SURVIVING DISCONNECTION: this happens automatically -- the block below
# re-execs the script under `setsid nohup ... & disown`, which puts it in a new
# session with no controlling terminal, so SIGHUP from a dropped SSH connection
# or a closed laptop never reaches it. stdin is /dev/null and output goes to
# run_nnunet_kfold.out. You do NOT need tmux, screen, or your own nohup.
#   check on it:  tail -f run_nnunet_kfold.out ; nvidia-smi ; pgrep -af nnUNetv2_train
#   stop it:      pkill -f nnUNetv2_train
# Note there is no scheduler on this node, so nothing will kill your job -- but
# equally nothing protects it from a reboot. Progress survives either way:
# re-running resumes from checkpoint_latest.pth.
#
# Safe to re-run: nnU-Net skips a fold whose checkpoint_final.pth already
# exists, and resumes an interrupted fold from checkpoint_latest.pth.

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${KFOLD_DETACHED:-}" ]; then
    LOG="$(pwd)/run_nnunet_kfold.out"
    echo "Detaching so this survives a dropped connection."
    echo "  progress (this script):  tail -f $LOG"
    echo "  progress (a given fold): tail -f logs_nnunet_fold*.log"
    echo "  GPU usage:               nvidia-smi"
    KFOLD_DETACHED=1 setsid nohup bash "$(pwd)/$(basename "$0")" "$@" </dev/null >"$LOG" 2>&1 &
    disown
    exit 0
fi

K_FOLDS=5
DATASET_ID=501
# ---- contention controls --------------------------------------------------
# This is a SHARED node with no scheduler: ~96 logical CPUs, and at the time of
# writing a load average of ~80 with 20 users and 9 of 10 GPUs already holding
# other people's memory. Nothing reserves anything for you, so the defaults
# below are deliberately polite rather than maximal.
#
# nnU-Net's own get_allowed_n_proc_DA() returns 12 here -- that is a DKFZ
# hostname lookup table falling through to its default, and it is PER FOLD. Five
# concurrent folds would mean 60 heavy 3D augmentation processes on top of an
# already-saturated node, which is what makes everything crawl. Cap it.
#
# Tuning: watch `nvidia-smi` and `uptime` after launch. GPU utilisation low AND
# load average high means the CPU is the bottleneck -- lower N_PROC_DA or
# MAX_CONCURRENT. GPU utilisation low and load average low means you can raise
# N_PROC_DA (the GPU is starving for augmented batches).
N_PROC_DA=${N_PROC_DA:-6}          # DA worker processes per fold
MAX_CONCURRENT=${MAX_CONCURRENT:-5} # folds training at once
MIN_FREE_MIB=${MIN_FREE_MIB:-16000} # skip GPUs with less free memory than this
MAX_UTIL_PCT=${MAX_UTIL_PCT:-25}    # skip GPUs busier than this
DRY_RUN=${DRY_RUN:-0}               # 1 = show the plan, launch nothing
DATASET_NAME=Dataset501_CardiacBoth
CONFIG=3d_fullres
TRAINER=nnUNetTrainer_250epochs   # 250 x 250 iters; nnU-Net's default is 1000 epochs

export nnUNet_raw=/hpc/jgeo610/nnunet/raw
export nnUNet_preprocessed=/hpc/jgeo610/nnunet/preprocessed
export nnUNet_results=/hpc/jgeo610/nnunet/results

# Overrides get_allowed_n_proc_DA()'s hostname lookup (it checks this first).
export nnUNet_n_proc_DA="$N_PROC_DA"
# Without these, every one of those DA workers spawns its own BLAS/OMP thread
# pool and N_PROC_DA processes silently become N_PROC_DA x cores threads. This
# single line is the difference between polite and pathological on a shared box.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NNUNET_ENV=/hpc/jgeo610/Virtual_ENV/nnunet_env
PYTHON="$NNUNET_ENV/bin/python"
TRAIN_CMD="$NNUNET_ENV/bin/nnUNetv2_train"

# ---- preflight ------------------------------------------------------------
# Five jobs that each die thirty seconds in is a worse failure than not
# starting, so everything gets checked up front.

[ -x "$PYTHON" ] || { echo "ERROR: no venv at $NNUNET_ENV. Create it with:"; \
    echo "  python3.10 -m venv $NNUNET_ENV"; \
    echo "  $NNUNET_ENV/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118"; \
    echo "  $NNUNET_ENV/bin/python -m pip install nnunetv2"; exit 1; }

"$PYTHON" -c "import nnunetv2" 2>/dev/null || {
    echo "ERROR: nnunetv2 not importable in $NNUNET_ENV"; exit 1; }

"$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: torch in $NNUNET_ENV cannot see a GPU"; exit 1; }

[ -x "$TRAIN_CMD" ] || { echo "ERROR: nnUNetv2_train not found at $TRAIN_CMD"; exit 1; }

PREP="$nnUNet_preprocessed/$DATASET_NAME"
[ -d "$PREP" ] || { echo "ERROR: $PREP missing. Run first:"; \
    echo "  nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -c $CONFIG"; exit 1; }

SPLITS="$PREP/splits_final.json"
[ -f "$SPLITS" ] || { echo "ERROR: $SPLITS missing -- nnU-Net would invent its own folds and"; \
    echo "       the comparison against the control would not pair. Run:"; \
    echo "  python make_nnunet_splits.py"; exit 1; }

# The whole comparison rests on these folds matching the control, so re-verify
# rather than trusting that make_nnunet_splits.py was run against the right runs.
"$PYTHON" - "$SPLITS" <<'PYEOF' || exit 1
import json, sys
splits = json.loads(open(sys.argv[1]).read())
val = [set(s['val']) for s in splits]
pool = set().union(*val)
assert len(splits) == 5, f'expected 5 folds, got {len(splits)}'
assert sum(len(v) for v in val) == len(pool) == 42, 'val folds must partition 42 patients'
for i, s in enumerate(splits):
    assert not (set(s['train']) & set(s['val'])), f'fold {i}: train/val overlap'
    assert set(s['train']) | set(s['val']) == pool, f'fold {i}: train+val != pool'
print(f'splits_final.json OK: 5 folds partitioning {len(pool)} patients')
PYEOF

# Explicit indices are taken as-is (you know what you are doing). Otherwise pick
# only GPUs that are actually free: enough spare memory for a 3d_fullres job AND
# not already busy with someone else's training. Blind round-robin over 0..9 is
# what lands a fold on top of a GPU at 100% utilisation.
if [ "$#" -gt 0 ]; then
    GPUS=("$@")
    echo "Using explicitly requested GPU(s): ${GPUS[*]}"
else
    mapfile -t GPUS < <(
        nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
                   --format=csv,noheader,nounits |
        awk -F', *' -v m="$MIN_FREE_MIB" -v u="$MAX_UTIL_PCT" \
            '$2 >= m && $3 <= u {print $3, -$2, $1}' |
        sort -n -k1,1 -k2,2 | awk '{print $3}')
    [ "${#GPUS[@]}" -gt 0 ] || { echo "ERROR: no GPU has >= ${MIN_FREE_MIB} MiB free and <= ${MAX_UTIL_PCT}% util."; \
        echo "       Check nvidia-smi, wait, or lower MIN_FREE_MIB / raise MAX_UTIL_PCT."; exit 1; }
    echo "Auto-selected idle GPU(s): ${GPUS[*]}  (>= ${MIN_FREE_MIB} MiB free, <= ${MAX_UTIL_PCT}% util)"
fi

N_GPUS="${#GPUS[@]}"
CONCURRENCY=$(( MAX_CONCURRENT < N_GPUS ? MAX_CONCURRENT : N_GPUS ))
echo "Dataset $DATASET_ID  config $CONFIG  trainer $TRAINER"
echo "Concurrency ${CONCURRENCY} fold(s), ${N_PROC_DA} DA workers each"
echo "  => up to $(( CONCURRENCY * (N_PROC_DA + 1) )) processes; node currently at load$(uptime | sed 's/.*load average//')"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | sed 's/^/  GPU /'

# ---- launch ---------------------------------------------------------------
RESULT_DIR="$nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__${CONFIG}"
# GPUs our own folds are on. Tracked by PID and pruned as folds finish, so a
# GPU is released for the next fold the moment its predecessor exits -- with
# only one or two GPUs free on this node, folds genuinely queue and must reuse
# them. A list that only ever grew would deadlock the tail of the run.
declare -A GPU_OF_PID=()
IN_USE=()
DRY_RESERVED=()    # dry run has no PIDs to track; reserve here so the plan is honest
refresh_in_use() {
    IN_USE=("${DRY_RESERVED[@]:-}")
    for _pid in "${!GPU_OF_PID[@]}"; do
        if kill -0 "$_pid" 2>/dev/null; then
            IN_USE+=("${GPU_OF_PID[$_pid]}")
        else
            unset 'GPU_OF_PID[$_pid]'
        fi
    done
}

for fold in $(seq 0 $((K_FOLDS - 1))); do
    if [ -f "$RESULT_DIR/fold_${fold}/checkpoint_final.pth" ]; then
        echo "fold ${fold}: already finished, skipping"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done

    # Re-pick a GPU per fold rather than assigning round-robin up front. On this
    # shared node availability moves fast -- five idle GPUs became two inside ten
    # minutes during testing -- so a list captured at launch is stale by fold 3.
    # Our own running folds are excluded explicitly: nnU-Net takes ~a minute to
    # allocate, so a just-launched fold still looks idle to nvidia-smi.
    if [ "$#" -gt 0 ]; then
        gpu="${GPUS[$((fold % N_GPUS))]}"
    else
        gpu=""
        for _try in 1 2 3 4 5 6 7 8 9 10 11 12; do
            refresh_in_use
            while read -r cand; do
                busy=0
                for used in "${IN_USE[@]:-}"; do [ "$used" = "$cand" ] && busy=1 && break; done
                [ "$busy" = "0" ] && gpu="$cand" && break
            done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
                                --format=csv,noheader,nounits |
                     awk -F', *' -v m="$MIN_FREE_MIB" -v u="$MAX_UTIL_PCT" \
                         '$2 >= m && $3 <= u {print $3, -$2, $1}' |
                     sort -n -k1,1 -k2,2 | awk '{print $3}')
            [ -n "$gpu" ] && break
            if [ "$DRY_RUN" = "1" ]; then
                echo "fold ${fold}: would wait here for a GPU to free up"
                break
            fi
            echo "fold ${fold}: no GPU free (>= ${MIN_FREE_MIB} MiB, <= ${MAX_UTIL_PCT}%), waiting 5 min..."
            sleep 300
        done
        if [ "$DRY_RUN" = "1" ]; then
            [ -z "$gpu" ] && continue
            # A dry run never frees a GPU, so reserve only up to CONCURRENCY of
            # them; beyond that the plan is "reuse as earlier folds finish".
            [ "${#DRY_RESERVED[@]}" -lt "$CONCURRENCY" ] && DRY_RESERVED+=("$gpu")
        fi
        [ -n "$gpu" ] || { echo "ERROR: fold ${fold} found no free GPU after an hour. Folds so far are still running; re-run later to finish the rest."; break; }
    fi
    resume=""
    [ -f "$RESULT_DIR/fold_${fold}/checkpoint_latest.pth" ] && resume="--c"
    echo "fold ${fold} -> GPU ${gpu} ${resume:+(resuming)}"
    if [ "$DRY_RUN" = "1" ]; then continue; fi

    CUDA_VISIBLE_DEVICES="$gpu" "$TRAIN_CMD" \
        "$DATASET_ID" "$CONFIG" "$fold" -tr "$TRAINER" $resume \
        > "logs_nnunet_fold${fold}.log" 2>&1 &
    GPU_OF_PID[$!]="$gpu"     # released by refresh_in_use when this fold exits

    sleep 5      # stagger, so the folds do not race on first cache reads
done

wait
if [ "$DRY_RUN" = "1" ]; then echo "DRY_RUN=1: nothing was launched."; exit 0; fi
echo "All folds done. Validation predictions are in:"
echo "  $RESULT_DIR/fold_<F>/validation/"
echo "Next: python score_nnunet_predictions.py   (in unet_env)"
