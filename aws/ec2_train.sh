#!/bin/bash
# Train the matching model on ALL training entities on an EC2 instance, then stop it.
#
# Run on an Amazon Linux 2023 instance whose IAM role can read/write the bucket:
#   sudo -i
#   curl -fsSL https://raw.githubusercontent.com/abbinavv/ml_challenge_2026/main/aws/ec2_train.sh -o ec2_train.sh
#   nohup bash ec2_train.sh <bucket> > run.log 2>&1 &
#   tail -f run.log
#
# Expects in s3://<bucket>/:  cache/{train_s1,train_s2,train_s3,train_cands_k30_plus}.parquet
#                             train_ground_truth.tsv
# Writes model to            s3://<bucket>/artifacts/v7/  (model.txt, stage1.txt, meta.json, train.log)
# Stops the instance at the end (shutdown behaviour must be "Stop").
set -euo pipefail
BUCKET=${1:?usage: ec2_train.sh <bucket> [n_train] [n_val]}
NTRAIN=${2:-2000000}
NVAL=${3:-150000}

echo "== $(date) installing tools"
dnf install -y -q git
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== $(date) fetching code"
rm -rf ber && git clone -q https://github.com/abbinavv/ml_challenge_2026.git ber && cd ber
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python \
  numpy==2.5.3 polars==1.44.2 pyarrow==25.0.1 scipy==1.18.1 scikit-learn==1.9.1 \
  lightgbm==4.7.0 rapidfuzz==3.14.6 anyascii==0.3.3 sparse-dot-topn==1.2.0

echo "== $(date) fetching data from s3://$BUCKET"
mkdir -p cache data/dataset/train artifacts
aws s3 cp "s3://$BUCKET/cache/" cache/ --recursive --only-show-errors
aws s3 cp "s3://$BUCKET/train_ground_truth.tsv" data/dataset/train/ --only-show-errors
export BER_DATA=data/dataset

echo "== $(date) training on up to $NTRAIN train + $NVAL validation entities"
.venv/bin/python -u src/train_model.py cache/train_cands_k30_plus.parquet artifacts/v7 \
  --full --n-train "$NTRAIN" --n-val "$NVAL" 2>&1 | grep -v "Warning" | tee artifacts/train_v7.log

echo "== $(date) uploading model"
cp artifacts/train_v7.log artifacts/v7/train.log
aws s3 cp artifacts/v7 "s3://$BUCKET/artifacts/v7/" --recursive --only-show-errors
echo "== $(date) done; stopping instance"
shutdown -h now
