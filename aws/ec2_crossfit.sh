#!/bin/bash
# Cross-fitted first stage: train on one half of the train entities, score the other
# half (out-of-sample probabilities for ALL train entities) and the test set, for both
# halves. Output feeds src/prior_stack.py with ~2.2M labelled entities instead of 150K.
#   nohup bash ec2_crossfit.sh <bucket> > run.log 2>&1 &
# Writes s3://<bucket>/crossfit/{cf0,cf1}/ (models, oof_*.parquet, test scored_pairs), then stops.
export HOME="${HOME:-/root}"
set -euo pipefail
BUCKET=${1:?usage: ec2_crossfit.sh <bucket>}
HALF=1100000
echo "== $(date) installing tools"
dnf install -y -q git
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
rm -rf ber && git clone -q https://github.com/abbinavv/ml_challenge_2026.git ber && cd ber
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python \
  numpy==2.5.3 polars==1.44.2 pyarrow==25.0.1 scipy==1.18.1 scikit-learn==1.9.1 \
  lightgbm==4.7.0 rapidfuzz==3.14.6 anyascii==0.3.3 sparse-dot-topn==1.2.0
mkdir -p cache data/dataset/train artifacts output
aws s3 cp "s3://$BUCKET/cache/" cache/ --recursive --only-show-errors
aws s3 cp "s3://$BUCKET/train_ground_truth.tsv" data/dataset/train/ --only-show-errors
export BER_DATA=data/dataset
for H in 0 1; do
  OFF=$(( H * HALF ))
  echo "== $(date) half $H: train on entities [$OFF, $OFF+$HALF)"
  .venv/bin/python -u src/train_model.py cache/train_cands_k30_plus.parquet "artifacts/cf$H" \
    --full --n-train 1050000 --n-val 50000 --offset "$OFF" --conflict-neg-weight 3 --max-rounds 5000 \
    2>&1 | grep --line-buffered -v "Warning" | tee "artifacts/cf$H.train.log"
  echo "== $(date) half $H: score the other half (out-of-sample)"
  .venv/bin/python -u src/score_validation.py cache/train_cands_k30_plus.parquet "artifacts/cf$H" "output/oof_cf$H.parquet" \
    --n-train "$HALF" --n-val 2000000 --offset "$OFF" 2>&1 | grep --line-buffered -v "Warning"
  echo "== $(date) half $H: score test"
  .venv/bin/python -u src/predict.py cache/test_cands_k30_fr_plus.parquet "artifacts/cf$H" "output/test_cf$H" \
    2>&1 | grep --line-buffered -v "Warning" | tail -3
  aws s3 cp "artifacts/cf$H" "s3://$BUCKET/crossfit/cf$H/model/" --recursive --only-show-errors
  aws s3 cp "output/oof_cf$H.parquet" "s3://$BUCKET/crossfit/cf$H/" --only-show-errors
  aws s3 cp "output/test_cf$H/scored_pairs.parquet" "s3://$BUCKET/crossfit/cf$H/test_scored_pairs.parquet" --only-show-errors
done
echo "== $(date) done; stopping instance"
shutdown -h now
