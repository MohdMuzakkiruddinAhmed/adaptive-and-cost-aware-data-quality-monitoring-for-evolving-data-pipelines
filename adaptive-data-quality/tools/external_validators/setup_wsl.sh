#!/usr/bin/env bash
# One-time set-up of the Deequ and TFDV sidecars in WSL (Ubuntu 24.04) or any Ubuntu/Debian Linux.
#
#   Windows (PowerShell, once):   wsl --install -d Ubuntu-24.04      then restart and create the Linux user
#   inside WSL:                    bash /mnt/c/Users/<you>/Desktop/DataQuality/adaptive-data-quality/tools/external_validators/setup_wsl.sh
#
# Two separate virtual environments, because the tools pin incompatible stacks:
#   ~/adq-tools/deequ   Java 17 + PySpark 3.5 + PyDeequ 1.7 (Deequ 2.0.21 for Spark 3.5, fetched from Maven Central)
#   ~/adq-tools/tfdv    TensorFlow Data Validation 1.21
# The paths match protocol.yaml external_tools.venvs.
set -euo pipefail

PYSPARK_VERSION="3.5.9"
PYDEEQU_VERSION="1.7.0"
TFDV_VERSION="1.21.0"
TOOLS="${HOME}/adq-tools"

echo "== system packages (Java 17, Python venv)"
sudo apt-get update -y
sudo apt-get install -y openjdk-17-jre-headless python3-venv python3-pip

mkdir -p "${TOOLS}"

echo "== Deequ environment"
python3 -m venv "${TOOLS}/deequ"
"${TOOLS}/deequ/bin/pip" install --upgrade pip
"${TOOLS}/deequ/bin/pip" install "pyspark==${PYSPARK_VERSION}" "pydeequ==${PYDEEQU_VERSION}" pandas pyarrow

echo "== TFDV environment"
python3 -m venv "${TOOLS}/tfdv"
"${TOOLS}/tfdv/bin/pip" install --upgrade pip
"${TOOLS}/tfdv/bin/pip" install "tensorflow-data-validation==${TFDV_VERSION}"

echo "== smoke test: Deequ (first start downloads the Deequ jar into ~/.ivy2)"
SPARK_VERSION=3.5 "${TOOLS}/deequ/bin/python" - <<'PY'
import json, pydeequ
from pyspark.sql import SparkSession
from pydeequ.suggestions import ConstraintSuggestionRunner, DEFAULT
spark = (SparkSession.builder.master("local[2]").config("spark.jars.packages", pydeequ.deequ_maven_coord)
         .config("spark.jars.excludes", pydeequ.f2j_maven_coord).config("spark.ui.enabled", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
df = spark.createDataFrame([(1.0, "a"), (2.0, "b"), (3.0, "a")], ["x", "c"])
b = ConstraintSuggestionRunner(spark).onData(df).addConstraintRule(DEFAULT())
res = b._ConstraintSuggestionRunBuilder.run()
sug = json.loads(spark._jvm.com.amazon.deequ.suggestions.ConstraintSuggestionResult.getConstraintSuggestionsAsJson(res))
print("Deequ OK:", pydeequ.deequ_maven_coord, "-", len(sug["constraint_suggestions"]), "suggestions on a toy table")
spark.stop()
PY

echo "== self-test of every Deequ call the sidecar uses (suggestions, row-level verification, anomaly checks)"
SPARK_VERSION=3.5 "${TOOLS}/deequ/bin/python" "$(dirname "$0")/run_deequ.py" --selftest --memory 2g

echo "== smoke test: TFDV"
"${TOOLS}/tfdv/bin/python" - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, tensorflow_data_validation as tfdv
s = tfdv.generate_statistics_from_dataframe(pd.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]}))
print("TFDV OK:", tfdv.__version__, "-", len(tfdv.infer_schema(s).feature), "features inferred")
PY

echo "== done. From Windows, run_all starts the tools automatically (protocol.yaml external_tools.runner: wsl)."
