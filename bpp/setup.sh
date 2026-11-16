#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/common.sh"
cd "$BPP_ROOT"
stage="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
env_profile="${BPP_ENV:-default}"
asset_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      if [[ $# -lt 2 ]]; then echo '--env requires default or sm120' >&2; exit 2; fi
      env_profile="$2"; shift 2 ;;
    --env=*) env_profile="${1#--env=}"; shift ;;
    *) asset_args+=("$1"); shift ;;
  esac
done
case "$env_profile" in
  default|sm120) ;;
  *) echo "Unknown environment: $env_profile (choose default or sm120)" >&2; exit 2 ;;
esac
case "$stage" in
  all|source|env|checkpoint|data|data-full|configure) ;;
  *) echo 'Usage: bash setup.sh [all|source|env|checkpoint|data|data-full|configure] [--env default|sm120] [--benchmark NAME] [--reuse-from PATH] [--local-only|--metadata-only]' >&2; exit 2 ;;
esac
python3 scripts/configure.py
if [[ "$stage" == all || "$stage" == source ]]; then
  python3 scripts/fetch_source.py --method "${BPP_SOURCE_METHOD:-auto}"
fi
if [[ "$stage" == all || "$stage" == env ]]; then
  python3 scripts/environment.py --env "$env_profile"
fi
if [[ "$stage" == all || "$stage" == checkpoint ]]; then
  python3 scripts/fetch_assets.py checkpoint "${asset_args[@]}"
fi
if [[ "$stage" == all || "$stage" == data ]]; then
  python3 scripts/fetch_assets.py data "${asset_args[@]}"
fi
if [[ "$stage" == data-full ]]; then
  python3 scripts/fetch_assets.py data --full "${asset_args[@]}"
fi
