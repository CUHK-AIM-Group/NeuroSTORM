#!/usr/bin/env bash
# Compatibility entry point; all project configuration lives in env.sh.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Run this file with: source ./env.sh\n' >&2
    exit 1
fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
