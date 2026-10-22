#!/usr/bin/env bash
# Checks MOD-001 to MOD-008 in docs/spec/models.md and AGT-001 to AGT-005 in
# docs/spec/agents.md, and prints one `sudus: REQ: pass|fail` line for each.
# Each requirement's tests in tests/spec_models.rs and tests/spec_agents.rs
# are named after it (mod_001_..., agt_001_...). A requirement passes only
# when at least one of its tests ran and all of them passed. When the tests
# do not build, every requirement fails.
set -uo pipefail
cd "$(dirname "$0")/.."

T=/home/shawn/.rustup/toolchains/1.95-x86_64-unknown-linux-gnu/bin
export PATH="$T:$PATH"
export CARGO_TARGET_DIR=/home/shawn/workspace2/scratchpads/targets/suprnova-ai-rs/w-spec
export RUSTC="$T/rustc" RUSTDOC="$T/rustdoc" RUSTUP_AUTO_INSTALL=0
cargo=("$T/cargo" test --locked --all-features --test spec_models --test spec_agents)
reqs=(MOD-001 MOD-002 MOD-003 MOD-004 MOD-005 MOD-006 MOD-007 MOD-008
  AGT-001 AGT-002 AGT-003 AGT-004 AGT-005)

failed=0
report() {
  if [ "$2" -eq 0 ]; then echo "sudus: $1: pass"; else echo "sudus: $1: fail"; failed=1; fi
}
# True when grep finds a match or cannot read a file; false only on a clean miss.
found() { grep "$@"; [ $? -ne 1 ]; }

# MOD-002 names that only the Vercel AI Gateway routing used.
gateway_names='canonical_provider_id|RoutingParams|RoutingTarget|ProviderRankingStrategy|GeoRegion|CloudRegion|routing_target'
gateway_free() {
  local params
  if found -rnE "$gateway_names" src tests README.md; then return 1; fi
  if found -n '`gateway`' README.md; then return 1; fi
  params=$(awk '/^pub struct InferenceRequestParams /,/^}/' src/models/params.rs)
  if [ -z "$params" ]; then
    echo "src/models/params.rs: InferenceRequestParams not found"
    return 1
  fi
  if printf '%s\n' "$params" | grep -nE '^\s*pub (routing|tags):'; then return 1; fi
  return 0
}

if ! "${cargo[@]}" --no-run; then
  for req in "${reqs[@]}"; do report "$req" 1; done
  exit 1
fi

for req in "${reqs[@]}"; do
  echo "== $req"
  filter="$(printf '%s' "$req" | tr 'A-Z-' 'a-z_')_"
  output=$("${cargo[@]}" -- "$filter" 2>&1)
  code=$?
  printf '%s\n' "$output"
  # Tests that ran, passed or failed, across every test binary's summary line.
  ran=$(printf '%s\n' "$output" |
    sed -n 's/^test result: [A-Za-z]*\. \([0-9][0-9]*\) passed; \([0-9][0-9]*\) failed.*/\1 \2/p' |
    awk '{ total += $1 + $2 } END { print total + 0 }')
  result=0
  if [ "$code" -ne 0 ]; then result=1; fi
  if [ "$ran" -eq 0 ]; then
    echo "$req: no test named $filter* ran"
    result=1
  fi
  if [ "$req" = MOD-002 ] && ! gateway_free; then result=1; fi
  report "$req" "$result"
done
exit "$failed"
