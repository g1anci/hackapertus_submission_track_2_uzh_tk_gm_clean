#!/usr/bin/env bash
# The SwissAdmin corpus can't be redistributed: we download it, run the notebooks and delete it after
set -euo pipefail
CORPUS=swissadmin_aligned_2014_05_26
URL=https://dialektkarten.ch/static/swissadmin/$CORPUS.zip

cleanup() { rm -rf "$CORPUS" /tmp/swissadmin /tmp/swissadmin.zip; }
trap cleanup EXIT
trap 'exit 130' INT TERM

curl -fL --retry 3 -o /tmp/swissadmin.zip "$URL"
unzip -q /tmp/swissadmin.zip -d /tmp/swissadmin
d=$(find /tmp/swissadmin -type d -name de-en -not -path '*/__MACOSX/*' -print -quit)
[ -n "$d" ] || { echo "unexpected zip layout: no de-en folder" >&2; exit 1; }
rm -rf "$CORPUS" && mv "$(dirname "$d")" "$CORPUS"

mkdir -p executed
papermill --log-output synthetic_data_1_preprocess.ipynb executed/1_preprocess.ipynb
cleanup   # only notebook 1 reads the raw corpus
papermill --log-output synthetic_data_2_train_encoder.ipynb executed/2_train_encoder.ipynb
papermill --log-output synthetic_data_3_create_data.ipynb executed/3_create_data.ipynb
