#!/bin/bash

KAPPA_MAX=5000.1
ERROR_FILE="error_neumann.csv"
THRESHOLD=0.01

# Créer le fichier avec l'entête s'il n'existe pas ou est vide
if [ ! -f "$ERROR_FILE" ] || [ ! -s "$ERROR_FILE" ]; then
    echo "N,degree,kappa/Pi,error_L2" > "$ERROR_FILE"
    echo "Created $ERROR_FILE with header."
fi

echo "Computations loading..."

for k in $(seq 100.1 100 $KAPPA_MAX); do

    N_START=$(awk "BEGIN {printf \"%d\", $10}")
    N_END=$(awk "BEGIN {printf \"%d\", $k * 0.9}")

    if [ "$N_START" -lt 10 ]; then
        N_START=10
    fi

    echo "=================================================="
    echo "Launching kappa_coeff = $k"
    echo "Adaptive N ranging from $N_START to $N_END"
    echo "=================================================="

    threshold_reached=0

    for (( n=N_START; n<=N_END; n+=1 )); do

        lines_before=$(wc -l < "$ERROR_FILE")

        ./build/release/main_exec $k $n

        lines_after=$(wc -l < "$ERROR_FILE")
        if [ "$lines_after" -le "$lines_before" ]; then
            echo "  --> WARNING: no new line written to CSV"
            continue
        fi

        last_error=$(tail -1 "$ERROR_FILE" | awk -F',' '{print $NF}')
        echo "  N=$n, error=$last_error"

        below=$(python3 -c "print(1 if float('$last_error') < $THRESHOLD else 0)" 2>/dev/null)

        if [ "${below}" = "1" ]; then
            echo "  --> Threshold reached at N=$n (error=$last_error)"
            threshold_reached=1
            break
        fi

    done

    if [ "$threshold_reached" -eq 0 ]; then
        echo "  --> WARNING: threshold NOT reached for kappa=$k"
    fi

done

echo "Done !"