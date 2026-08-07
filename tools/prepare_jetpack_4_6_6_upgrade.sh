#!/usr/bin/env bash

set -euo pipefail

readonly TARGET_L4T_REPOSITORY="r32.7"
readonly EXPECTED_CURRENT_L4T_PREFIX="# R32 (release), REVISION: 5.1,"
readonly NVIDIA_SOURCE_FILE="/etc/apt/sources.list.d/nvidia-l4t-apt-source.list"
readonly PROXY_CONFIGURATION_FILE="/etc/apt/apt.conf.d/90jetracer-upgrade-proxy"
readonly STATE_DIRECTORY="/var/lib/jetracer-upgrade"
readonly PROXY_URL="${JETRACER_UPGRADE_PROXY_URL:-http://192.168.50.3:8899}"
readonly MIN_UNUSED_CAPACITY_BYTES=1073741824

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this script with sudo." >&2
    exit 2
  fi
}

root_partition() {
  findmnt -n -o SOURCE /
}

parent_device() {
  local partition="$1"
  local parent
  parent="$(lsblk -n -o PKNAME "${partition}" | tr -d '[:space:]')"
  if [[ -z "${parent}" ]]; then
    echo "Cannot determine the parent device for ${partition}." >&2
    exit 3
  fi
  printf '/dev/%s\n' "${parent}"
}

partition_number() {
  local partition="$1"
  if [[ ! "${partition}" =~ ^/dev/mmcblk[0-9]+p([0-9]+)$ ]]; then
    echo "Cannot determine the partition number for ${partition}." >&2
    exit 3
  fi
  printf '%s\n' "${BASH_REMATCH[1]}"
}

capture_state() {
  local device="$1"
  install -d -m 0700 "${STATE_DIRECTORY}"
  if [[ ! -f "${STATE_DIRECTORY}/partition-table.before.sfdisk" ]]; then
    sfdisk --dump "${device}" \
      >"${STATE_DIRECTORY}/partition-table.before.sfdisk"
  fi
  if [[ ! -f "${STATE_DIRECTORY}/nvidia-l4t-apt-source.list.before" ]]; then
    cp -a "${NVIDIA_SOURCE_FILE}" \
      "${STATE_DIRECTORY}/nvidia-l4t-apt-source.list.before"
  fi
  if [[ ! -f "${STATE_DIRECTORY}/packages.before.tsv" ]]; then
    dpkg-query -W -f='${binary:Package}\t${Version}\n' \
      >"${STATE_DIRECTORY}/packages.before.tsv"
  fi
  cp /etc/nv_tegra_release "${STATE_DIRECTORY}/nv_tegra_release.before"
}

expand_root_partition() {
  local partition="$1"
  local device="$2"
  local number="$3"
  local partition_bytes_before device_bytes partition_bytes_after

  partition_bytes_before="$(blockdev --getsize64 "${partition}")"
  device_bytes="$(blockdev --getsize64 "${device}")"
  echo "Root partition before expansion: ${partition_bytes_before} bytes"
  echo "Parent device capacity: ${device_bytes} bytes"

  if (( partition_bytes_before + MIN_UNUSED_CAPACITY_BYTES < device_bytes )); then
    # The Waveshare image was created for a much smaller card, so its backup
    # GPT header is still at the old end of disk. Move that header first, then
    # preserve partition 1's start sector while extending only its end.
    sgdisk --move-second-header "${device}"
    sgdisk --verify "${device}"
    printf ',+\n' | sfdisk --no-reread --no-tell-kernel \
      --partno "${number}" "${device}"
    sync
    partprobe "${device}" || true
    udevadm settle
  fi

  partition_bytes_after="$(blockdev --getsize64 "${partition}")"
  if (( partition_bytes_after == partition_bytes_before && \
        partition_bytes_before + MIN_UNUSED_CAPACITY_BYTES < device_bytes )); then
    echo "The partition table was expanded, but the running kernel has not" >&2
    echo "re-read it. Reboot, then run this script again." >&2
    exit 10
  fi

  e2fsck -fn "${partition}" || true
  resize2fs "${partition}"
  df -h /
}

configure_upgrade_repositories() {
  if ! grep -q 'repo.download.nvidia.com/jetson' "${NVIDIA_SOURCE_FILE}"; then
    echo "NVIDIA Jetson repository configuration is missing." >&2
    exit 4
  fi
  sed -i -E \
    "s#(repo\.download\.nvidia\.com/jetson/[^[:space:]]+[[:space:]]+)r32\.[0-9]+#\\1${TARGET_L4T_REPOSITORY}#g" \
    "${NVIDIA_SOURCE_FILE}"

  cat >"${PROXY_CONFIGURATION_FILE}" <<EOF
Acquire::http::Proxy "${PROXY_URL}";
Acquire::https::Proxy "${PROXY_URL}";
EOF

  echo "Configured NVIDIA repositories:"
  grep -E '^[[:space:]]*deb .*repo.download.nvidia.com/jetson' \
    "${NVIDIA_SOURCE_FILE}"
}

simulate_upgrade() {
  apt-get update
  apt-get --simulate dist-upgrade \
    | tee "${STATE_DIRECTORY}/dist-upgrade.simulation.txt"
  echo
  echo "Simulation saved to ${STATE_DIRECTORY}/dist-upgrade.simulation.txt"
  echo "No packages were upgraded by this script."
}

main() {
  require_root

  local release root device number
  release="$(head -n 1 /etc/nv_tegra_release)"
  if [[ "${release}" != "${EXPECTED_CURRENT_L4T_PREFIX}"* && \
        "${release}" != "# R32 (release), REVISION: 7."* ]]; then
    echo "Unexpected L4T release: ${release}" >&2
    exit 5
  fi

  root="$(root_partition)"
  if [[ "${root}" != /dev/mmcblk*p1 ]]; then
    echo "Refusing to resize unexpected root partition: ${root}" >&2
    exit 6
  fi
  device="$(parent_device "${root}")"
  number="$(partition_number "${root}")"
  if [[ "${number}" != "1" ]]; then
    echo "Refusing to resize unexpected root partition number: ${number}" >&2
    exit 6
  fi

  capture_state "${device}"
  expand_root_partition "${root}" "${device}" "${number}"
  configure_upgrade_repositories
  simulate_upgrade
}

main "$@"
