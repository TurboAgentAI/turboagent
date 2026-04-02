#!/bin/bash
set -e

# Build the serve command from environment variables
CMD="python -m turboagent.cli serve --model ${TURBOAGENT_MODEL} --host 0.0.0.0 --port 8000"

[ -n "$TURBOAGENT_BACKEND" ] && CMD="$CMD --backend $TURBOAGENT_BACKEND"
[ -n "$TURBOAGENT_KV_MODE" ] && CMD="$CMD --kv-mode $TURBOAGENT_KV_MODE"
[ -n "$TURBOAGENT_CONTEXT" ] && CMD="$CMD --context $TURBOAGENT_CONTEXT"
[ -n "$TURBOAGENT_API_KEYS" ] && CMD="$CMD --api-keys $TURBOAGENT_API_KEYS"
[ -n "$TURBOAGENT_RATE_LIMIT" ] && CMD="$CMD --rate-limit $TURBOAGENT_RATE_LIMIT"

echo "Starting: $CMD"
exec $CMD
