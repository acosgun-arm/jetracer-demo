#!/usr/bin/env bash

set -euo pipefail

readonly STATE_DIRECTORY="/var/lib/jetracer-upgrade"
readonly LOG_FILE="${STATE_DIRECTORY}/dist-upgrade.apply.log"
readonly NVIDIA_SOURCE_FILE="/etc/apt/sources.list.d/nvidia-l4t-apt-source.list"
readonly REQUIRED_FREE_BYTES=10737418240
readonly TARGET_L4T_VERSION_PREFIX="32.7.6-"
readonly TARGET_JETPACK_VERSION_PREFIX="4.6.6-"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this script with sudo." >&2
    exit 2
  fi
}

preflight() {
  local root_free_bytes
  if ! grep -Eq \
    'repo\.download\.nvidia\.com/jetson/(common|t210)[[:space:]]+r32\.7' \
    "${NVIDIA_SOURCE_FILE}"; then
    echo "The NVIDIA repositories are not configured for r32.7." >&2
    exit 3
  fi
  if [[ ! -f /etc/apt/apt.conf.d/90jetracer-upgrade-proxy ]]; then
    echo "The temporary upgrade proxy is not configured." >&2
    exit 4
  fi
  root_free_bytes="$(df --output=avail -B1 / | tail -n 1 | tr -d '[:space:]')"
  if (( root_free_bytes < REQUIRED_FREE_BYTES )); then
    echo "At least ${REQUIRED_FREE_BYTES} free bytes are required." >&2
    exit 5
  fi
  if [[ -n "$(dpkg --audit)" ]]; then
    echo "dpkg reports an incomplete package transaction." >&2
    dpkg --audit >&2
    exit 6
  fi
}

main() {
  require_root
  install -d -m 0700 "${STATE_DIRECTORY}"
  exec > >(tee -a "${LOG_FILE}") 2>&1

  echo "JetPack upgrade started at $(date -Is)"
  preflight

  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get -y --download-only dist-upgrade
  apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confnew" \
    dist-upgrade

  dpkg --configure -a
  apt-get -y --fix-broken install
  # NVIDIA documents that a JetPack 4.5 -> 4.6 dist-upgrade can remove or
  # omit JetPack components. Reinstalling the meta-package completes the SDK.
  apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confnew" \
    install nvidia-jetpack
  dpkg --configure -a
  apt-get -y --fix-broken install
  dpkg --audit

  local l4t_version jetpack_version
  l4t_version="$(dpkg-query -W -f='${Version}' nvidia-l4t-core)"
  jetpack_version="$(dpkg-query -W -f='${Version}' nvidia-jetpack)"
  if [[ "${l4t_version}" != "${TARGET_L4T_VERSION_PREFIX}"* ]]; then
    echo "Unexpected nvidia-l4t-core version: ${l4t_version}" >&2
    exit 7
  fi
  if [[ "${jetpack_version}" != "${TARGET_JETPACK_VERSION_PREFIX}"* ]]; then
    echo "Unexpected nvidia-jetpack version: ${jetpack_version}" >&2
    exit 8
  fi
  sync

  echo "JetPack upgrade package transaction completed at $(date -Is)"
  echo "A reboot is required; this script does not reboot automatically."
}

main "$@"
