#!/bin/bash

KAPPA_MAX=1000.1
KAPPA_MIN=200.1
KAPPA_STEP=10
ERROR_FILE="error_ipdg_robin.csv"
DEGREE=2
GAMMA=$(awk "BEGIN {print 2 * ($DEGREE + 1) * ($DEGREE + 1)}")  # IPDG penalty
THRESHOLD=0.01

# Créer le fichier avec l'entête s'il n'existe pas ou est vide
if [ ! -f "$ERROR_FILE" ] || [ ! -s "$ERROR_FILE" ]; then
    echo "N,degree,kappa/Pi,gamma,error_energy,error_L2" > "$ERROR_FILE"
    echo "Created $ERROR_FILE with header."
fi

echo "Computations loading..."

for k in $(seq $KAPPA_MIN $KAPPA_STEP $KAPPA_MAX); do

    # N_START=$(awk "BEGIN {printf \"%d\", $k * 0.7}")
    # N_END=$(awk "BEGIN {printf \"%d\", $k * 1.7}")

    # Fit : N = 22.1 * kappa - 424, avec marge 0.85 / 1.3
    # N_START=$(awk "BEGIN {n = int(22.1 * $k - 424); printf \"%d\", (n * 0.85 < 10) ? 10 : int(n * 0.85)}")
    # N_END=$(awk "BEGIN {printf \"%d\", int((22.1 * $k - 424) * 1.3)}")
    N_START=$(awk "BEGIN {printf \"%d\", $k * 2.8}")
    N_END=$(awk "BEGIN {printf \"%d\", $k * 10.0}")

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

        ./build/release/main_exec $k $n ipdg_robin $DEGREE $GAMMA

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