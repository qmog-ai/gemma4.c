#!/usr/bin/env bash
set -euo pipefail

make benchmark EXTRA_CFLAGS=-fno-inline

control_dir=$(mktemp -d)
trap 'rm -rf "$control_dir"' EXIT

mkfifo "$control_dir/control" "$control_dir/ack"
exec {control_fd}<>"$control_dir/control"
exec {ack_fd}<>"$control_dir/ack"

PERF_CTL_FD=$control_fd PERF_ACK_FD=$ack_fd perf record -o perf.data -D -1 -m 64 -e cycles:pp --control fd:$control_fd,$ack_fd -- ./benchmark "$@"
