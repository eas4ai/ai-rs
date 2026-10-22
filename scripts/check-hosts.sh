#!/usr/bin/env bash
# Checks HST-001 to HST-007 in docs/spec/hosts.md through
# scripts/spec-tests.sh, with the tests in tests/spec_hosts.rs. The tests
# replay recorded HTTP traffic through a scripted transport, so no request
# leaves the machine and no key is needed.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/spec-tests.sh

tests=(spec_hosts)
reqs=(HST-001 HST-002 HST-003 HST-004 HST-005 HST-006 HST-007)

run_spec_tests
