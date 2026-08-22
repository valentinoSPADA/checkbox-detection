#!/bin/bash
# Runs the detection sidecar and the API as one container's foreground process.
#
# bash, not sh: the die-together rule below rests on `wait -n`, which is a bash builtin. On
# Debian /bin/sh is dash, where `wait -n` is a syntax error at the moment the first process
# exits -- a failure that would only ever appear in production, and only during an incident.
#
# Two processes in one container needs a supervisor, and the choice here is a 40-line shell
# script rather than supervisord or s6. The reason is what a supervisor is *for*: keeping a
# process alive when its sibling dies. That is precisely the wrong behaviour on a platform
# that already restarts unhealthy machines -- an API left running after the sidecar has
# crashed answers every request with a 503 while reporting itself as up, which is the failure
# that takes an afternoon to find. So this script does the opposite: if either process exits,
# the container exits, and the platform restarts it with both.
#
# set -e is not enough for that, because these run in the background. The wait loop below is.

set -eu

log() { echo "{\"level\":\"INFO\",\"src\":\"entrypoint\",\"msg\":\"$1\"}"; }

# Fly (and most PaaS) inject PORT. ADDR is what the Go service reads, so the two are joined
# here rather than in the service -- the binary should not have to know which platform it is
# running on.
ADDR=":${PORT:-8080}"
export ADDR

log "starting detection sidecar"
# A single worker on purpose. Detection is CPU-bound and holds ~290 MB at its peak, so a
# second worker would double the memory ceiling to serve requests that would be competing for
# the same core anyway. Concurrency belongs at the machine level, not inside one container.
python -m uvicorn app.main:app \
    --host 127.0.0.1 --port 8000 \
    --workers 1 --log-level warning &
DETECTOR_PID=$!

log "starting api"
/usr/local/bin/api &
API_PID=$!

# Forward termination to both children so a rolling deploy lets an in-flight page finish
# rather than killing it. Without this, only the shell receives the signal and the platform
# ends up SIGKILLing whatever was still working.
terminate() {
    log "shutdown signal received"
    kill -TERM "$DETECTOR_PID" "$API_PID" 2>/dev/null || true
    wait "$DETECTOR_PID" "$API_PID" 2>/dev/null || true
    exit 0
}
trap terminate TERM INT

# Waits for whichever child exits FIRST, then leaves. `wait -n` is what makes the
# die-together rule real; a plain `wait` would block until both had exited and would happily
# sit on a half-dead container in the meantime.
wait -n "$DETECTOR_PID" "$API_PID"
STATUS=$?
log "a process exited with status ${STATUS}; stopping the container so it restarts whole"
kill -TERM "$DETECTOR_PID" "$API_PID" 2>/dev/null || true
exit "$STATUS"
