#!/usr/bin/env bash
# Checks OAI-001 and OAI-002 in docs/spec/openai.md through
# scripts/spec-tests.sh, with the tests in tests/spec_openai.rs. The tests
# build or record each request the crate sends and answer it with a reply
# written in the test, so no request leaves the machine and no key is needed.
# A pass shows the crate sends what the AI SDK for TypeScript 7.0.148 sends,
# not that OpenAI still accepts it.
set -uo pipefail
cd "$(dirname "$0")/.."
. scripts/spec-tests.sh

tests=(spec_openai)
reqs=(OAI-001 OAI-002)

run_spec_tests
