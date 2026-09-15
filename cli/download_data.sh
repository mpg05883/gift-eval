#!/bin/bash
set -e

LOCAL_DIR="./data"
MAX_ATTEMPTS=5

attempt=1
until hf download Salesforce/GiftEval \
    --repo-type=dataset \
    --local-dir "$LOCAL_DIR"; do
    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        echo "hf download failed after $MAX_ATTEMPTS attempts" >&2
        exit 1
    fi
    echo "hf download failed (attempt $attempt/$MAX_ATTEMPTS), retrying in 10s..." >&2
    attempt=$((attempt + 1))
    sleep 10
done

grep -qxF "GIFT_EVAL=\"$LOCAL_DIR\"" .env 2>/dev/null || echo "GIFT_EVAL=\"../../$LOCAL_DIR\"" >> .env