#!/bin/bash -eu

cd "$SRC/astra-synthetic-gating-demo"

python3 -m pip install -r requirements.txt
python3 -m pip install .

for fuzzer in $(find "$SRC" -name 'fuzz_*.py'); do
  compile_python_fuzzer "$fuzzer"
done