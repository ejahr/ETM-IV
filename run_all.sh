#!/bin/bash

cd ~/icec_project/icec_HeNe
#source ~/icec_project/icec/bin/activate
source venv/bin/activate

scripts=("HeNe-BA.py" "HeNe-AB.py" "HeNe-BX.py" "HeNe-XB.py")
scripts=("HeNe-AB.py")

echo "========== start =========="

for script in "${scripts[@]}"
do
    echo "===== $script ====="
    start_time=$(date +%s)
    python "$script"
    end_time=$(date +%s)
    elapsed_seconds=$((end_time - start_time))

    elapsed_formatted=$(printf "%02d:%02d:%02d" \
        $((elapsed_seconds / 3600)) \
        $(((elapsed_seconds % 3600) / 60)) \
        $((elapsed_seconds % 60)))

    echo "Elapsed time for $script: $elapsed_formatted"
    echo "" 
done

echo "========== finish =========="
