#!/usr/bin/env bash

set -euo pipefail

readonly TEST_FILE="${JETRACER_SD_TEST_FILE:-/var/tmp/jetracer-sd-validation.bin}"
readonly TEST_SIZE_BYTES="${JETRACER_SD_TEST_SIZE_BYTES:-8589934592}"
readonly BLOCK_SIZE_BYTES="${JETRACER_SD_TEST_BLOCK_SIZE_BYTES:-16777216}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 2
fi
if [[ ! "${TEST_SIZE_BYTES}" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "${BLOCK_SIZE_BYTES}" =~ ^[1-9][0-9]*$ ]] || \
   (( TEST_SIZE_BYTES % BLOCK_SIZE_BYTES != 0 )); then
  echo "Test size must be a positive multiple of the block size." >&2
  exit 3
fi
if [[ -e "${TEST_FILE}" ]]; then
  echo "Refusing to overwrite existing test file: ${TEST_FILE}" >&2
  exit 4
fi

readonly BLOCK_COUNT=$((TEST_SIZE_BYTES / BLOCK_SIZE_BYTES))

echo "Writing ${TEST_SIZE_BYTES} bytes to ${TEST_FILE}"
dd if=/dev/zero of="${TEST_FILE}" bs="${BLOCK_SIZE_BYTES}" \
  count="${BLOCK_COUNT}" oflag=direct conv=fdatasync status=progress

echo "Performing a direct read"
dd if="${TEST_FILE}" of=/dev/null bs="${BLOCK_SIZE_BYTES}" \
  iflag=direct status=progress

echo "Verifying every byte"
cmp --bytes="${TEST_SIZE_BYTES}" "${TEST_FILE}" /dev/zero

rm -- "${TEST_FILE}"
sync
echo "SD write/read verification passed"
