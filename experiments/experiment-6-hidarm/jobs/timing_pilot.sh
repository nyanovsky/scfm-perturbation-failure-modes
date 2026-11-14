#!/usr/bin/env bash
# Timing pilot: ONE model, ONE line, FULL coverage, wall clock and peak RSS
# recorded. README section 12 item 6, and the number every later wave is scaled
# against.
#
#   jobs/timing_pilot.sh            scgpt / hepg2
#   jobs/timing_pilot.sh scprint jurkat
#
# scGPT/HepG2 by default: the cheapest model on the smaller line, so the pilot
# costs ~20 min rather than ~2.5 h, and TranscriptFormer's per-gene cost is
# already measured (17.4 s/gene on the 8-gene smoke) and scales from it.
#
# /usr/bin/time -v is here for ONE number that no other run reports: peak
# resident set. The main pass's projected accumulator is 34 GB (HepG2) to 55 GB
# (Jurkat) and is memory-mapped for that reason (config.STORE_MEMMAPPED); peak
# RSS is the only evidence that the mapping actually kept it out of RAM.
source "$(dirname "$0")/_common.sh"

MODEL=${1:-scgpt}
LINE=${2:-hepg2}

for stage in basis main; do
  echo "=== timing pilot $stage $MODEL/$LINE  $(date -Is)"
  PYTHONPATH=src /usr/bin/time -v "$(py "$MODEL")" -m scfm_hidarm.extract.run \
      --model "$MODEL" --cell-line "$LINE" --stage "$stage" --device cuda
  record "timing $stage $MODEL/$LINE" $?
done
finish "timing pilot $MODEL/$LINE"
