#!/bin/bash
# Submit chesslesson training to Condor (login node only).
set -euo pipefail

REPO=$HOME/verl-agent
BID="${1:-50}"

mkdir -p "${REPO}/condor_out/output" "${REPO}/condor_out/error" "${REPO}/condor_out/log"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARN: WANDB_API_KEY is not set; wandb logging may fail." >&2
fi

condor_submit_bid "${BID}" "${REPO}/recipe/hgpo/chesslesson_train.sub"
echo "Monitor: condor_q ; tail -f ${REPO}/condor_out/output/chesslesson_train.*.out"
