#!/bin/sh
# Re-score every model's own predicted response with the fold-mean correction.
#
#   sh $SILICO_EXPERIMENT_RELATIVE_DIR/jobs/direct_fix.sh
#
# run 3's ``score_run`` takes a prediction of the RESIDUAL and forms the raw
# prediction as argument + training-fold mean. The model predicts the perturbed
# profile itself, so the earlier call added that mean on top of a prediction
# that already contained it and the raw cell was not a correlation with the
# measured effect. ``score_direct`` now removes the fold mean before the call,
# which makes all three cells of the grid mean what they say.
#
# Nothing is recomputed from a model here: the predicted profiles were saved by
# each training run, so this job needs only those, each run's fold assignment,
# and the two arrays run 3's scorer reads. It still wants a GPU because run 3's
# LineBundle puts its tensors on the device it is given.
set -eu
cd "$SILICO_EXPERIMENT_RELATIVE_DIR"
sh jobs/env_gpu.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
STAGED="$SILICO_INPUT_ARTIFACTS_DIR/experiments/$SILICO_EXPERIMENT_ID"
BUNDLE="$SILICO_INPUT_ARTIFACTS_DIR/experiments/exp_01m0emwktvfe892x6ebqyga5ws/bundle"
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/direct_fix"
mkdir -p "$OUT"

echo "=== staged bundle ==="
ls -R "$BUNDLE" | head -20
echo "=== checkpoint dirs ==="
ls "$STAGED/checkpoints"

echo
echo "=== re-scoring the direct prediction, both reference points ==="
python3 -m scfm_ft.followup direct \
    --bundle "$BUNDLE" --checkpoints "$STAGED/checkpoints" --out "$OUT" --device cuda

echo
echo "=== what came out ==="
ls -la "$OUT"
python3 - <<'PY'
import json, os, pathlib
import pandas as pd

out = pathlib.Path(os.environ["SILICO_EXPERIMENT_ARTIFACTS_DIR"]) / "direct_fix"
rows = json.loads((out / "direct_rescored.json").read_text())
frame = pd.DataFrame(rows)
keep = frame[frame.baseline == "measured_control_mean"]
print(keep[["cell_line", "arm", "raw", "disattenuated", "centered"]].to_string(index=False))
print(f"\n{len(frame)} rows over {frame.run.nunique()} runs and {frame.baseline.nunique()} baselines")
PY

tar -czf "$SILICO_EXPERIMENT_ARTIFACTS_DIR/direct_fix.tar.gz" -C "$OUT" .
echo "packed $(du -sh "$SILICO_EXPERIMENT_ARTIFACTS_DIR/direct_fix.tar.gz" | cut -f1)"
