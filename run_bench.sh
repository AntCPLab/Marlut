#!/bin/bash
set -e

# Default to using default features
# BUILD_MODE="default"

# Parse build mode flag
# for arg in "$@"; do
#     if [ "$arg" == "--default" ]; then
#         BUILD_MODE="default"
#     elif [ "$arg" == "--non-default" ]; then
#         BUILD_MODE="non-default"
#     fi
# done

BUILD_MODE=1 # 0: default, 1: non-default

echo "[BUILD] Building project..."

if [ "$BUILD_MODE" -eq 1 ]; then
   cargo build --release --no-default-features --features "clmul GF8_3 GF8NonPre_3 FP16Exp_3 FP8Add_3 Boolean_3"
else
   cargo build --release
fi

echo "[BUILD] Build finished."

BANDWIDTH=320mbit
DELAY_MS=20ms

ENABLE_TC=1

# -------------------------------
# Traffic Control (tc) functions
# -------------------------------
setup_tc() {
    echo "[TC] Setting up traffic control..."

    IFACE=$(ip -o link show | awk -F': ' '!/lo/ {print $2; exit}')

    if [ -z "$IFACE" ]; then
        echo "[TC] No valid network interface found!"
        return 1
    fi

    echo "[TC] Using interface: $IFACE"

    sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
    # sudo tc qdisc add dev "$IFACE" root handle 1: tbf rate "$BANDWIDTH" burst 1000m latency 4000ms
    # sudo tc qdisc add dev "$IFACE" parent 1: handle 10: netem delay "$DELAY_MS"
    sudo tc qdisc add dev "$IFACE" root netem rate "$BANDWIDTH" delay "$DELAY_MS"

    echo "[TC] Current tc qdisc:"
    sudo tc qdisc show dev "$IFACE"
    echo "[TC] TC setup completed successfully!"
}

teardown_tc() {
    echo "[TC] Cleaning up traffic control..."
    IFACE=$(ip -o link show | awk -F': ' '!/lo/ {print $2; exit}')
    if [ -n "$IFACE" ]; then
        sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
        echo "[TC] TC rules removed."
    else
        echo "[TC] No valid interface found to clean."
    fi
}

# -------------------------------
# --- Main Script Logic ---
# -------------------------------

if [ -z "$1" ]; then
    echo "Usage: $0 <config_file_name>"
    echo "Example: $0 p1.toml"
    exit 1
fi

CONFIG_FILE="$1"
shift  

# for arg in "$@"; do
#     if [ "$arg" == "--no-tc" ]; then
#         ENABLE_TC=0
#         echo "Traffic control will NOT be applied."
#     fi
# done

# Fixed parameters
protocols=(
    "lut256-sp"
    # "lut256-sp-fp16-exp"
    # "mal-lut256-sp"
    # "mal-lut256-sp-fp16-exp"
    # "maepro10"
    # "lut256-sp-fp8-add"
    # "mal-lut256-sp-fp8-add"
    # "lut256-sp-gf8"
    # "lut256-sp-boolean"
    # "mal-lut256-sp-boolean"
    # "ip-benchmark"
    # "lut16"
    # "mal-lut16-bitstring"
    # "mal-lut16-ohv"
)
simds=(
    "262144"
    # "1048576"
    # "4194304"
    # "16777216"
)
RAYON_THREADS=1
THREADS=1
REP=4
TARGET_BIN="target/release/marlut"
BASE_CSV_NAME="result"

echo "--- Preparing to run with configuration file: $CONFIG_FILE ---"

# --- TC Setup ---
if [ "$ENABLE_TC" -eq 1 ]; then
    setup_tc
    if [ $? -ne 0 ]; then
        echo "Aborting script due to TC setup failure."
        exit 1
    fi
fi
# ----------------

# Iterate over all combinations of protocols and simd values
for protocol in "${protocols[@]}"; do
    for simd in "${simds[@]}"; do
        CONFIG_BASE=$(basename "$CONFIG_FILE" .toml)
        OUTPUT_CSV="${BASE_CSV_NAME}_${CONFIG_BASE}_${protocol}_${simd}.csv"
        
        echo "--- Running combination:"
        echo "    Configuration file: $CONFIG_FILE"
        echo "    Protocol:      $protocol"
        echo "    SIMD value:    $simd"
        echo "    Output file:   $OUTPUT_CSV"
        echo "---"
        
        COMMAND="RAYON_NUM_THREADS=$RAYON_THREADS $TARGET_BIN --config $CONFIG_FILE --threads $THREADS --simd $simd --rep $REP --csv $OUTPUT_CSV $protocol"
        echo "Executing command: $COMMAND"
        
        eval "$COMMAND"

        if [ $? -eq 0 ]; then
            echo " Command executed successfully! Results saved to $OUTPUT_CSV"
        else
            echo " Command execution failed! Please check the error."
        fi

        echo ""
    done
done

# --- TC Teardown ---
if [ "$ENABLE_TC" -eq 1 ]; then
    teardown_tc
fi
# -------------------

echo "All commands executed for $CONFIG_FILE!"
