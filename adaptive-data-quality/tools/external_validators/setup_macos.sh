#!/usr/bin/env bash
# One-time set-up of the Deequ and TFDV sidecars on macOS (the counterpart of setup_wsl.sh).
#
#   Deequ  runs natively: Homebrew OpenJDK 17 + a venv at ~/adq-tools/deequ (PySpark 3.5 + PyDeequ 1.7)
#   TFDV   runs in Docker: image adq-tfdv:1.21.0 (Ubuntu 24.04, linux/amd64), built from Dockerfile.tfdv
#
# Prerequisites:  brew install openjdk@17 python@3.11      and Docker Desktop running
# Usage:          bash tools/external_validators/setup_macos.sh [deequ|tfdv|all]     (default: all)
# The paths match protocol.yaml external_tools (runner, java_home, venvs, docker).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PYSPARK_VERSION="3.5.9"
PYDEEQU_VERSION="1.7.0"
TFDV_VERSION="1.21.0"
TOOLS="${HOME}/adq-tools"
JAVA_HOME_17="${JAVA_HOME_17:-/opt/homebrew/opt/openjdk@17}"
DEEQU_PYTHON="${DEEQU_PYTHON:-/opt/homebrew/opt/python@3.11/bin/python3.11}"   # PySpark 3.5 supports up to 3.11/3.12
TFDV_IMAGE="${TFDV_IMAGE:-adq-tfdv:${TFDV_VERSION}}"
WHAT="${1:-all}"

setup_deequ() {
  echo "== Deequ: Java 17 at ${JAVA_HOME_17}"
  [ -x "${JAVA_HOME_17}/bin/java" ] || { echo "missing: brew install openjdk@17"; exit 1; }
  "${JAVA_HOME_17}/bin/java" -version
  [ -x "${DEEQU_PYTHON}" ] || { echo "missing: brew install python@3.11 (or set DEEQU_PYTHON)"; exit 1; }

  echo "== Deequ environment (${TOOLS}/deequ, $("${DEEQU_PYTHON}" --version))"
  mkdir -p "${TOOLS}"
  "${DEEQU_PYTHON}" -m venv "${TOOLS}/deequ"
  "${TOOLS}/deequ/bin/pip" install --upgrade pip
  "${TOOLS}/deequ/bin/pip" install "pyspark==${PYSPARK_VERSION}" "pydeequ==${PYDEEQU_VERSION}" pandas pyarrow

  echo "== self-test of every Deequ call the sidecar uses (first start downloads the Deequ jar into ~/.ivy2)"
  JAVA_HOME="${JAVA_HOME_17}" SPARK_VERSION=3.5 SPARK_LOCAL_IP=127.0.0.1 \
    "${TOOLS}/deequ/bin/python" "${HERE}/run_deequ.py" --selftest --memory 2g
}

setup_tfdv() {
  echo "== TFDV: Docker image ${TFDV_IMAGE} (linux/amd64)"
  docker info > /dev/null 2>&1 || { echo "Docker is not running: start Docker Desktop"; exit 1; }
  docker build --platform linux/amd64 --build-arg "TFDV_VERSION=${TFDV_VERSION}" \
    -t "${TFDV_IMAGE}" -f "${HERE}/Dockerfile.tfdv" "${HERE}"

  echo "== smoke test: TFDV"
  docker run --rm --platform linux/amd64 "${TFDV_IMAGE}" /opt/adq-tools/tfdv/bin/python -c '
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, tensorflow_data_validation as tfdv
s = tfdv.generate_statistics_from_dataframe(pd.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]}))
print("TFDV OK:", tfdv.__version__, "-", len(tfdv.infer_schema(s).feature), "features inferred")'
}

case "${WHAT}" in
  deequ) setup_deequ ;;
  tfdv)  setup_tfdv ;;
  all)   setup_deequ; setup_tfdv ;;
  *)     echo "usage: $0 [deequ|tfdv|all]"; exit 2 ;;
esac
echo "== done (${WHAT}). run_all starts the tools itself (protocol.yaml external_tools.runner)."
