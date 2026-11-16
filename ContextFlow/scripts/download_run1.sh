#!/bin/bash
# Download the remaining ContextFlow_run1/19999 checkpoint files from Google Drive
# via aria2c multi-connection (gdown was ~0.55MB/s; aria2c -x8 ~3.4MB/s).
# Resumable: re-run to continue interrupted files (.aria2 control files).

CKPT=/data2/huangxiangyun/project/icl/ContextFlow/checkpoints/ContextFlow/ContextFlow_run1/19999
PROXY=http://127.0.0.1:50311
BASE="https://drive.usercontent.google.com/download?export=download&confirm=t"
LOGDIR=/data2/huangxiangyun/project/icl/ContextFlow/logs/aria2
mkdir -p "$LOGDIR" "$CKPT"

# id|relpath
FILES="
1V0QkuYPuLv9COTcoxEujk6AuhFNpBHYp|params/ocdbt.process_0/d/0982d5892eec8655f759a64c532c80f1
1AJ50PP_k-D4DzPNo0wRiwr60S7wOKCWd|params/ocdbt.process_0/d/4888ef41461d55726f7dbe7973597032
1tzT8FBewMgb5Z_2EKcMV0M0Ianch92MX|params/ocdbt.process_0/d/9195b8140e2283508ab1d98cb4557ebe
1beDYwg6UjTdvg1ovg2dI-jaJT9O2_bfe|params/ocdbt.process_0/d/6743752b99d9270a19041349ecd1cef6
12DSwyW5OmLCP0t9UhdNbA1rGgdXgbZn9|params/ocdbt.process_0/d/9587076fc2613372d0fde0996430241d
1iHrnBdtdASLjCjCDYqT9DXGJ4ek-WPGL|params/ocdbt.process_0/d/d103ff03208852161c800ffa992f01d4
1R965rr9XcYB14ReUz8XmfeXt0QtTkmW4|params/ocdbt.process_0/d/dc37dde22e6cceb122622e0ff38d0f54
1mE6UYiDNQw0ppObetfVnT-FAo20WmKHy|params/ocdbt.process_0/d/de7304fe50f591c0230f845fb8f87c0b
1MZlSUuBI9XTXcJY4RLIZJLBgcZ5n0TI0|params/ocdbt.process_0/d/e4da0a2b9ecdc50064fb59d9c1237863
168WvGlzFzR_WhsC3Yb__wjB0Q4wzpwCM|params/ocdbt.process_0/d/e15b047215d17e2f5ca4731506584622
1QFwRHWtM3Jlt-zi1abxyDtdGraAaMg8L|params/ocdbt.process_0/d/f51965b3a8044f767642a7adcf9e83ee
1t8zhEjXInNtJ4j_sH-Mfy_hFLWljLL3b|params/ocdbt.process_0/manifest.ocdbt
1FH_6bTl-0vYCdYgCqcZxN_9B9GPc8Hkr|params/_METADATA
1ZbJ6JT-K78f2Mo_j-vWOBh2k0AyKT9BL|params/_sharding
1VtKEacQE3PoYH1pdzZUaD7UhGVFsSjC0|params/manifest.ocdbt
1YRlhpXcxnE1gOckpHP2BuysFf_-RgXFt|_CHECKPOINT_METADATA
1UV71ia2NVD20CkXkPJgXtFf4DrdLno7B|assets/physical-intelligence/libero/norm_stats.json
"

export CKPT PROXY BASE LOGDIR
echo "$FILES" | xargs -P3 -I{} bash -c '
    IFS="|" read -r id rel <<< "{}"
    [ -z "$id" ] && exit 0
    out="$CKPT/$rel"
    mkdir -p "$(dirname "$out")"
    if [ -f "$out" ] && [ ! -f "$out.aria2" ]; then
        echo "SKIP $rel"; exit 0
    fi
    name=$(basename "$rel")
    aria2c -x8 -s8 -k1M --file-allocation=none --console-log-level=warn \
        --summary-interval=30 --all-proxy="$PROXY" --auto-file-renaming=false \
        --allow-overwrite=true -c \
        "$BASE&id=$id" -o "$name" -d "$(dirname "$out")" \
        > "$LOGDIR/$name.log" 2>&1
    echo "EXIT $? $rel"
'
echo "=== done ==="
find "$CKPT" \( -name "*.aria2" -o -name "*.part" \)
