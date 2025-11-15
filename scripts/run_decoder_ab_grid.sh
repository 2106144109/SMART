#!/usr/bin/env bash
set -euo pipefail

# Small A/B grid for decoder rotation settings on maritime:
# - Always enable position-difference angle
# - Sweep rotation sign in {pos,neg}
# - Sweep rotation offset in {0,90,-90,180}
# - Uniform sample 200 scenes, do not save maps

cd "$(dirname "$0")/.."

# Resolve checkpoint
if [[ -n "${CKPT:-}" && -f "$CKPT" ]]; then
  : # use provided CKPT
else
  if ls -t logs/maritime_ft0deg_ckpts/*.ckpt >/dev/null 2>&1; then
    CKPT="$(ls -t logs/maritime_ft0deg_ckpts/*.ckpt | head -n1)"
  elif ls -t logs/maritime_checkpoints/*.ckpt >/dev/null 2>&1; then
    CKPT="$(ls -t logs/maritime_checkpoints/*.ckpt | head -n1)"
  else
    echo "❌ No checkpoint found. Set CKPT=/path/to/ckpt.ckpt and retry." >&2
    exit 1
  fi
fi

echo "Using CKPT: $CKPT"

NUM=200
OUTDIR_BASE="folium_pred_maps_abgrid_${NUM}"
mkdir -p "$OUTDIR_BASE"

SIGNS=(pos neg)
OFFS=(0 90 -90 180)

for sign in "${SIGNS[@]}"; do
  for deg in "${OFFS[@]}"; do
    tag="sign_${sign}_off_${deg}"
    log="${OUTDIR_BASE}/metrics_${tag}.log"
    echo "\n=== Running ${tag} ==="
    DECODER_USE_DIFF_ANGLE=1 \
    DECODER_ROT_SIGN="$sign" \
    DECODER_ROT_OFFSET_DEG="$deg" \
    FOLIUM_USE_REF_ANCHOR=1 \
    FOLIUM_DISABLE_AUTO_AXIS=1 \
    FOLIUM_SAMPLE_MODE=uniform \
    python visualize_predictions_folium.py \
      --config configs/train/train_maritime.yaml \
      --pretrain_ckpt "$CKPT" \
      --split test --num_scenes "$NUM" \
      --no_save_map | tee "$log"
  done
done

echo "\n=== Summary (ADE/FDE avg and DirCos avg) ==="
for sign in "${SIGNS[@]}"; do
  for deg in "${OFFS[@]}"; do
    tag="sign_${sign}_off_${deg}"
    log="${OUTDIR_BASE}/metrics_${tag}.log"
    printf "%s  " "$tag"
    # ADE/FDE averages
    awk -F '[= m,]+' '/Scene .*ADE=/{ade+=$3; fde+=$5; n++} END{if(n>0) printf "ADE=%.2f m FDE=%.2f m  ", ade/n, fde/n; else printf "ADE=NA FDE=NA  "}' "$log"
    # DirCos average
    awk -F ': ' '/DirCos/{d+=$NF; n++} END{if(n>0) printf "DirCos=%.3f\n", d/n; else printf "DirCos=NA\n"}' "$log"
  done
done

echo "\nDone. Logs are under: $OUTDIR_BASE"


