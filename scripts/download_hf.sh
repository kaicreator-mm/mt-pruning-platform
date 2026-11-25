#!/usr/bin/env bash
# Resumable download of the pinned Hy-MT3-30B-A3B revision straight from huggingface.co.
# usage: download_hf.sh <manifest.tsv> <snapshot_dir> [parallel_downloads]
#   manifest.tsv lines: <size>\t<oid>\t<path>   (oid = sha256 for LFS files, git-blob sha1 otherwise)
# Each file is verified against its oid before it is marked OK.
set -u
MANIFEST=$1; DEST=$2; PAR=${3:-3}
REPO=tencent/Hy-MT3-30B-A3B
REV=08ae9acea91abe265c6236e757e49d8851e1ce06
BASE="https://huggingface.co/$REPO/resolve/$REV"
LOG="$DEST/../download.log"
STATE="$DEST/../download.state"
mkdir -p "$DEST"; : > "$STATE"

fsize() { stat -c %s "$1" 2>/dev/null || echo 0; }

fetch_one() {
    local size=$1 oid=$2 path=$3 out="$DEST/$3" cur rc attempt
    mkdir -p "$(dirname "$out")"
    for attempt in $(seq 1 500); do
        cur=$(fsize "$out")
        if [ "$cur" -gt "$size" ]; then rm -f "$out"; cur=0; fi
        if [ "$cur" -eq "$size" ]; then
            if [ ${#oid} -eq 64 ]; then
                if [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$oid" ]; then
                    echo "OK $path" >> "$STATE"; return 0
                fi
                echo "$(date -u +%FT%TZ) SHA_MISMATCH redo $path" >> "$LOG"; rm -f "$out"
            else
                if [ "$(git hash-object "$out")" = "$oid" ]; then
                    echo "OK $path" >> "$STATE"; return 0
                fi
                echo "$(date -u +%FT%TZ) BLOB_MISMATCH redo $path" >> "$LOG"; rm -f "$out"
            fi
        fi
        curl -sS -L -C - --connect-timeout 30 --speed-limit 2048 --speed-time 90 \
             -o "$out" "$BASE/$path" 2>> "$LOG"; rc=$?
        cur=$(fsize "$out")
        [ "$cur" -eq "$size" ] && continue   # verify on the next pass
        echo "$(date -u +%FT%TZ) retry=$attempt rc=$rc have=$cur/$size $path" >> "$LOG"
        sleep 5
    done
    echo "FAILED $path" >> "$STATE"; return 1
}

export -f fetch_one fsize
export DEST BASE LOG STATE

# small files first (sequentially), then shards in parallel
awk -F'\t' '$1 < 100000000' "$MANIFEST" | while IFS=$'\t' read -r size oid path; do fetch_one "$size" "$oid" "$path"; done
awk -F'\t' '$1 >= 100000000' "$MANIFEST" | xargs -P "$PAR" -d '\n' -I{} bash -c 'IFS=$(printf "\t") read -r s o p <<< "{}"; fetch_one "$s" "$o" "$p"'

want=$(awk -F'\t' '{n++} END{print n}' "$MANIFEST")
have=$(grep -c '^OK' "$STATE")
if [ "$have" -eq "$want" ]; then
    date -u +%FT%TZ > "$DEST/../DOWNLOAD_COMPLETE"; echo "ALL_OK $have/$want" >> "$STATE"
else
    echo "INCOMPLETE $have/$want" >> "$STATE"
fi
