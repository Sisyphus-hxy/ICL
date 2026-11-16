#!/bin/bash
# One worker: policy server on <cvd_gpu>:<port> + rollout client rendering
# on <phys_gpu>. NOTE the two different GPU numberings (see README):
#   cvd_gpu  = CUDA_VISIBLE_DEVICES value for the JAX server (5/6=A6000,
#              1-4=6000D, 0=Blackwell)
#   phys_gpu = nvidia-smi physical index for the client's EGL context
# Usage: bash pi05_eval/worker.sh <mode> <cvd_gpu> <phys_gpu> <port> [client args...]
# Example:
#   bash pi05_eval/worker.sh contextflow 5 0 8000 all
#   bash pi05_eval/worker.sh liberogen   2 3 8002 liberogen_spatial_combination
set -u
cd "$(dirname "$0")/.."
source pi05_eval/env.sh

MODE="${1:?mode: contextflow|liberogen}"
CVD="${2:?CUDA_VISIBLE_DEVICES value for server}"
PHYS="${3:?physical nvidia-smi index for client EGL}"
PORT="${4:?port}"
shift 4

STAMP=$(date +%Y%m%d_%H%M%S)
SLOG="logs/serve_${MODE}_${CVD}_${PORT}_${STAMP}.log"
CLOG="logs/client_${MODE}_${CVD}_${PORT}_${STAMP}.log"
mkdir -p logs

# --- start policy server ----------------------------------------------------
echo "[worker] server: CVD=$CVD port=$PORT log=$SLOG"
(cd "$OPENPI" && CUDA_VISIBLE_DEVICES="$CVD" exec uv run scripts/serve_policy.py --env LIBERO --port "$PORT") \
    >"$SLOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

# --- wait for server readiness ---------------------------------------------
echo "[worker] waiting for server on :$PORT ..."
for i in $(seq 1 600); do
    if ! kill -0 $SPID 2>/dev/null; then
        echo "[worker] server died; see $SLOG"; exit 1
    fi
    if "$OPENPI/.venv/bin/python" - "$PORT" <<'PY' 2>/dev/null
import socket, sys
s = socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1)
s.close()
PY
    then
        break
    fi
    sleep 5
done
echo "[worker] server up after ~$((i*5))s; starting client (log=$CLOG)"

# --- run rollout client ------------------------------------------------------
if [ "$MODE" = contextflow ]; then
    bash pi05_eval/run_contextflow.sh "$PHYS" "$PORT" "$@" >"$CLOG" 2>&1
elif [ "$MODE" = liberogen ]; then
    bash pi05_eval/run_liberogen.sh "$PHYS" "$PORT" "$@" >"$CLOG" 2>&1
else
    echo "[worker] unknown mode $MODE"; exit 2
fi
RC=$?
echo "[worker] client exited rc=$RC; stopping server"
kill $SPID 2>/dev/null
wait $SPID 2>/dev/null
exit $RC
