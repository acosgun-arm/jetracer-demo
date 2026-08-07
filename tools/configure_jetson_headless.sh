#!/usr/bin/env bash

set -euo pipefail

readonly TARGET_USER="${JETRACER_HEADLESS_USER:-jetson}"
readonly SUDOERS_FILE="/etc/sudoers.d/90-jetracer-headless"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 2
fi

if ! id "${TARGET_USER}" >/dev/null 2>&1; then
  echo "Target user does not exist: ${TARGET_USER}" >&2
  exit 3
fi

wifi_connection="$(
  nmcli -t -f NAME,TYPE,DEVICE connection show --active \
    | awk -F: '$2 == "802-11-wireless" && $3 == "wlan0" {print $1; exit}'
)"
if [[ -z "${wifi_connection}" ]]; then
  echo "No active Wi-Fi connection exists on wlan0." >&2
  exit 4
fi

nmcli connection modify "${wifi_connection}" \
  connection.autoconnect yes \
  connection.permissions ""

systemctl enable ssh
systemctl restart ssh

printf '%s ALL=(ALL:ALL) NOPASSWD: ALL\n' "${TARGET_USER}" \
  >"${SUDOERS_FILE}"
chmod 0440 "${SUDOERS_FILE}"
visudo -cf "${SUDOERS_FILE}"

echo "Configured Wi-Fi connection: ${wifi_connection}"
echo "SSH is enabled and active."
echo "Passwordless sudo is enabled for: ${TARGET_USER}"
