#!/usr/bin/env bash
# Checks HST-001 to HST-032 in docs/spec/hosts.md through
# scripts/spec-tests.sh, with the tests in tests/spec_hosts.rs. The tests
# record each request the crate sends and answer it with a reply written in
# the test, so no request leaves the machine and no key is needed. A pass
# shows the crate sends what the AI SDK for TypeScript 7.0.148 sends, not that
# a host still accepts it.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/spec-tests.sh

tests=(spec_hosts)
reqs=(HST-001 HST-002 HST-003 HST-004 HST-005 HST-006 HST-007 HST-008 HST-009
  HST-010 HST-011 HST-012 HST-013 HST-014 HST-015 HST-016 HST-017 HST-018
  HST-019 HST-020 HST-021 HST-022 HST-023 HST-024 HST-025 HST-026 HST-027
  HST-028 HST-029 HST-030 HST-031 HST-032)

run_spec_tests
