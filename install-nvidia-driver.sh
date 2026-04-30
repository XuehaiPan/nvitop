#!/bin/bash
# ==============================================================================
#
# Usage: bash install-nvidia-driver.sh [--package=PKG] [--upgrade-only] [--latest]
#                                      [--proprietary | --open] [--dry-run] [--yes] [--help]
#
# Examples:
#
#     bash install-nvidia-driver.sh
#     bash install-nvidia-driver.sh --package=nvidia-driver-595
#     bash install-nvidia-driver.sh --package=nvidia-driver-595-open
#     bash install-nvidia-driver.sh --upgrade-only
#     bash install-nvidia-driver.sh --latest
#     bash install-nvidia-driver.sh --latest --open
#
# ==============================================================================
# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
#
# Copyright 2021-2025 Xuehai Pan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# shellcheck disable=SC2016,SC2310,SC2312

set -u
set -e
shopt -s inherit_errexit &>/dev/null || true

function abort() {
	echo "$@" >&2
	exit 1
}

# shellcheck disable=2292
if [ -z "${BASH_VERSION:-}" ]; then
	abort "Bash is required to interpret this script."
fi

export LANGUAGE="C:en"

# shellcheck disable=1091
if [[ "$(uname -s)" != "Linux" ]] || (source /etc/os-release && [[ "${NAME:-}" != "Ubuntu" ]]); then
	abort "This script only supports Ubuntu Linux."
elif [[ -n "${WSL_DISTRO_NAME:-}" ]] || (uname -r | grep -qiF 'microsoft'); then
	abort "Please install the NVIDIA driver in the Windows host system rather than in WSL."
fi

# String formatters
if [[ -t 1 ]]; then
	tty_escape() { printf "\033[%sm" "$1"; }
else
	tty_escape() { :; }
fi
tty_mkbold() { tty_escape "1;$1"; }
tty_blue="$(tty_mkbold 34)"
tty_red="$(tty_mkbold 31)"
tty_green="$(tty_mkbold 32)"
tty_yellow="$(tty_mkbold 33)"
tty_white="$(tty_mkbold 37)"
tty_bold="$(tty_mkbold 39)"
tty_dim="$(tty_escape 90)"
tty_reset="$(tty_escape 0)"

function usage() {
	cat <<EOS
Usage: bash $(basename "$0") [--package=PKG] [--upgrade-only] [--latest]
                             [--proprietary | --open] [--dry-run] [--yes] [--help]

Options:
  --package PKG    Install the specified driver package. (e.g. ${tty_bold}nvidia-driver-595${tty_reset} or ${tty_bold}nvidia-driver-595-open${tty_reset})
  --upgrade-only   Keep the installed NVIDIA driver package and only upgrade the package version.
  --latest         Upgrade to the latest NVIDIA driver, the old driver package may be removed.
  --proprietary    Prefer the proprietary kernel-module flavor (e.g. ${tty_bold}nvidia-driver-595${tty_reset}).
  --open           Prefer the open kernel-module flavor (e.g. ${tty_bold}nvidia-driver-595-open${tty_reset}).
                   If neither flag is set, the installed driver's flavor is preserved
                   (proprietary on fresh hosts).
  --dry-run, -n    List all available NVIDIA driver packages and exit.
  --yes, -y        Do not ask for confirmation.
  --help, -h       Show this help and exit.

Examples:

    bash $(basename "$0")
    bash $(basename "$0") --package=nvidia-driver-595
    bash $(basename "$0") --package=nvidia-driver-595-open
    bash $(basename "$0") --upgrade-only
    bash $(basename "$0") --latest
    bash $(basename "$0") --latest --open

${tty_yellow}NOTE:${tty_reset} During the install process, the script will offload the NVIDIA kernel modules. Please terminate the
processes that are using the NVIDIA GPUs, e.g., \`watch nvidia-smi\`, \`nvitop\`, and the GNOME Display Manager.
The script will install the latest NVIDIA driver if no NVIDIA driver is installed in the system.
EOS
}

### Parse the command line arguments ###############################################################

REQUESTED_DRIVER=''
UPGRADE_ONLY=''
LATEST=''
FLAVOR=''          # '', 'proprietary', or 'open'
FLAVOR_EXPLICIT='' # set when the user explicitly passed --proprietary or --open
DRY_RUN=''
YES=''
unset HAVE_SUDO_ACCESS
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--package)
			REQUESTED_DRIVER="$1"
			shift
			;;
		--package=*)
			REQUESTED_DRIVER="${arg#*=}"
			;;
		--upgrade-only)
			if [[ -n "${LATEST}" ]]; then
				abort 'Both option `--upgrade-only` and `--latest` are set.'
			fi
			UPGRADE_ONLY=1
			;;
		--latest)
			if [[ -n "${UPGRADE_ONLY}" ]]; then
				abort 'Both option `--upgrade-only` and `--latest` are set.'
			fi
			LATEST=1
			;;
		--proprietary)
			if [[ "${FLAVOR}" == 'open' ]]; then
				abort 'Both option `--proprietary` and `--open` are set.'
			fi
			FLAVOR='proprietary'
			FLAVOR_EXPLICIT=1
			;;
		--open)
			if [[ "${FLAVOR}" == 'proprietary' ]]; then
				abort 'Both option `--proprietary` and `--open` are set.'
			fi
			FLAVOR='open'
			FLAVOR_EXPLICIT=1
			;;
		--dry-run | -n)
			DRY_RUN=1
			HAVE_SUDO_ACCESS=1
			;;
		--yes | -y)
			YES=1
			;;
		--help | -h)
			usage
			exit
			;;
		*)
			usage >&2
			echo >&2
			abort "Invalid option '${arg}'"
			;;
	esac
done

# Default flavor when neither --proprietary nor --open is given.
: "${FLAVOR:=proprietary}"

### Functions ######################################################################################

function apt-list-packages() {
	dpkg-query --show --showformat='${Installed-Size} ${binary:Package} ${Version} ${Status}\n' |
		grep -v deinstall | sort -n | awk '{ print $1" "$2" "$3 }'
}

function apt-list-nvidia-packages() {
	local packages
	packages="$(
		apt-list-packages |
			awk '$2 ~ /nvidia.*-([0-9]+)(-open)?(:.*)?$/ { print $2 }' |
			sort
	)"
	echo "${packages//$'\n'/ }"
}

function apt-installed-version() {
	apt-cache policy "$1" | grep -F 'Installed' | awk '{ print $2 }'
}

function driver-major() {
	# nvidia-driver-595      -> 595
	# nvidia-driver-595-open -> 595
	local d="${1#nvidia-driver-}"
	echo "${d%-open}"
}

function driver-flavor() {
	# nvidia-driver-595      -> proprietary
	# nvidia-driver-595-open -> open
	if [[ "$1" == *-open ]]; then
		echo 'open'
	else
		echo 'proprietary'
	fi
}

function toggle-open-suffix() {
	# nvidia-driver-595      -> nvidia-driver-595-open
	# nvidia-driver-595-open -> nvidia-driver-595
	if [[ "$1" == *-open ]]; then
		echo "${1%-open}"
	else
		echo "$1-open"
	fi
}

function driver-available() {
	# Predicate: is "$1" present in the global AVAILABLE_DRIVERS array?
	local d
	for d in "${AVAILABLE_DRIVERS[@]}"; do
		if [[ "${d}" == "$1" ]]; then
			return 0
		fi
	done
	return 1
}

function apt-candidate-version() {
	apt-cache policy "$1" | grep -F 'Candidate' | awk '{ print $2 }'
}

# From https://github.com/XuehaiPan/Dev-Setup.git
function exec_cmd() {
	printf "%s" "$@" | awk \
		'BEGIN {
			RESET = "\033[0m";
			BOLD = "\033[1m";
			UNDERLINE = "\033[4m";
			UNDERLINEOFF = "\033[24m";
			RED = "\033[31m";
			GREEN = "\033[32m";
			YELLOW = "\033[33m";
			WHITE = "\033[37m";
			GRAY = "\033[90m";
			IDENTIFIER = "[_a-zA-Z][_a-zA-Z0-9]*";
			idx = 0;
			in_string = 0;
			double_quoted = 1;
			printf("%s$", RESET BOLD WHITE);
		}
		{
			for (i = 1; i <= NF; ++i) {
				style = WHITE;
				post_style = WHITE;
				if (!in_string) {
					if ($i ~ /^-/)
						style = YELLOW;
					else if ($i == "sudo" && idx == 0) {
						style = UNDERLINE GREEN;
						post_style = UNDERLINEOFF WHITE;
					}
					else if ($i ~ "^" IDENTIFIER "=" && idx == 0) {
						style = GRAY;
						'"if (\$i ~ \"^\" IDENTIFIER \"=[\\\"']\") {"'
							in_string = 1;
							double_quoted = ($i ~ "^" IDENTIFIER "=\"");
						}
					}
					else if ($i ~ /^[12&]?>>?/ || $i == "\\")
						style = RED;
					else {
						++idx;
						'"if (\$i ~ /^[\"']/) {"'
							in_string = 1;
							double_quoted = ($i ~ /^"/);
						}
						if (idx == 1)
							style = GREEN;
					}
				}
				if (in_string) {
					if (style == WHITE)
						style = "";
					post_style = "";
					'"if ((double_quoted && \$i ~ /\";?\$/ && \$i !~ /\\\\\";?\$/) || (!double_quoted && \$i ~ /';?\$/))"'
						in_string = 0;
				}
				if (($i ~ /;$/ && $i !~ /\\;$/) || $i == "|" || $i == "||" || $i == "&&") {
					if (!in_string) {
						idx = 0;
						if ($i !~ /;$/)
							style = RED;
					}
				}
				if ($i ~ /;$/ && $i !~ /\\;$/)
					printf(" %s%s%s;%s", style, substr($i, 1, length($i) - 1), (in_string ? WHITE : RED), post_style);
				else
					printf(" %s%s%s", style, $i, post_style);
				if ($i == "\\")
					printf("\n\t");
			}
		}
		END {
			printf("%s\n", RESET);
		}' >&2
	# shellcheck disable=SC2294
	eval "$@"
}

# From https://github.com/Homebrew/install.git
shell_join() {
	local arg
	printf "%s" "$1"
	shift
	for arg in "$@"; do
		printf " "
		printf "%s" "${arg// /\ }"
	done
}

chomp() {
	printf "%s" "${1/"$'\n'"/}"
}

ohai() {
	printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$(shell_join "$@")"
}

warn() {
	printf "${tty_red}Warning:${tty_reset} %s\n" "$(chomp "$1")"
}

ring_bell() {
	# Use the shell's audible bell.
	if [[ -t 1 ]]; then
		printf "\a"
	fi
}

getc() {
	local save_state
	save_state="$(/bin/stty -g)"
	/bin/stty raw -echo
	IFS='' read -r -n 1 -d '' "$@"
	/bin/stty "${save_state}"
}

CONFIRM_MESSAGE="Press ${tty_bold}RETURN${tty_reset}/${tty_bold}ENTER${tty_reset} to continue or any other key to abort:"
wait_for_user() {
	local c
	echo
	echo "${CONFIRM_MESSAGE}"
	getc c
	# we test for \r and \n because some stuff does \r instead
	if ! [[ "${c}" == $'\r' || "${c}" == $'\n' ]]; then
		exit 1
	fi
}

function have_sudo_access() {
	if [[ "${EUID:-"${UID}"}" == "0" ]]; then
		return 0
	fi

	if [[ ! -x "/usr/bin/sudo" ]]; then
		return 1
	fi

	local -a SUDO=("/usr/bin/sudo")
	if [[ -n "${SUDO_ASKPASS-}" ]]; then
		SUDO+=("-A")
	fi

	if [[ -z "${HAVE_SUDO_ACCESS-}" ]]; then
		ohai "Checking sudo access (press ${tty_yellow}Ctrl+C${tty_white} to list the driver versions only)." >&2
		exec_cmd "${SUDO[*]} -v && ${SUDO[*]} -l mkdir &>/dev/null"
		HAVE_SUDO_ACCESS="$?"
	fi

	return "${HAVE_SUDO_ACCESS}"
}

### Add Graphics Drivers PPA #######################################################################

# shellcheck disable=SC2310
if have_sudo_access; then
	exec_cmd 'sudo apt-get update'
	if ! (apt-cache policy | grep -qF 'graphics-drivers/ppa/ubuntu'); then
		exec_cmd 'sudo apt-get install software-properties-common apt-transport-https --yes'
		exec_cmd 'sudo add-apt-repository ppa:graphics-drivers/ppa --yes'
		exec_cmd 'sudo apt-get update'
	fi
	echo
else
	DRY_RUN=1
fi

### Query and list available driver packages #######################################################

# shellcheck disable=SC2207
AVAILABLE_DRIVERS=($(
	apt-cache search --names-only nvidia-driver |
		awk '$1 ~ /^nvidia-driver-([0-9]+)(-open)?$/ { print $1 }' |
		sort -V
))

if [[ "${#AVAILABLE_DRIVERS[@]}" -eq 0 ]]; then
	abort "No available drivers found from APT."
fi

# `sort -V` orders `nvidia-driver-595` before `nvidia-driver-595-open`, so picking
# `${AVAILABLE_DRIVERS[-1]}` would silently flip the default to open. Pick per-flavor instead.
LATEST_PROPRIETARY_DRIVER=''
LATEST_OPEN_DRIVER=''
for driver in "${AVAILABLE_DRIVERS[@]}"; do
	if [[ "$(driver-flavor "${driver}")" == 'open' ]]; then
		LATEST_OPEN_DRIVER="${driver}"
	else
		LATEST_PROPRIETARY_DRIVER="${driver}"
	fi
done

INSTALLED_DRIVER="$(apt-list-packages | awk '$2 ~ /nvidia-driver-([0-9]+)(-open)?$/ { print $2 }')"
if [[ -n "${INSTALLED_DRIVER}" ]]; then
	INSTALLED_DRIVER_VERSION="$(apt-installed-version "${INSTALLED_DRIVER}")"
	INSTALLED_DRIVER_CANDIDATE_VERSION="$(apt-candidate-version "${INSTALLED_DRIVER}")"
	INSTALLED_FLAVOR="$(driver-flavor "${INSTALLED_DRIVER}")"
else
	INSTALLED_DRIVER_VERSION=''
	INSTALLED_DRIVER_CANDIDATE_VERSION=''
	INSTALLED_FLAVOR=''
fi

# Effective flavor for picking LATEST_DRIVER: explicit `--proprietary`/`--open` wins; otherwise
# preserve the installed driver's flavor; fresh hosts fall through to the `FLAVOR` default.
EFFECTIVE_FLAVOR="${FLAVOR}"
if [[ -z "${FLAVOR_EXPLICIT}" && -n "${INSTALLED_FLAVOR}" ]]; then
	EFFECTIVE_FLAVOR="${INSTALLED_FLAVOR}"
fi

if [[ "${EFFECTIVE_FLAVOR}" == 'open' ]]; then
	LATEST_DRIVER="${LATEST_OPEN_DRIVER}"
	LATEST_FALLBACK="${LATEST_PROPRIETARY_DRIVER}"
else
	LATEST_DRIVER="${LATEST_PROPRIETARY_DRIVER}"
	LATEST_FALLBACK="${LATEST_OPEN_DRIVER}"
fi
# Resolution-time fallback only matters when the script needs LATEST_DRIVER to compute
# REQUESTED_DRIVER. An explicit `--package=PKG` is the user's authoritative override and is
# allowed through even when the preserved flavor has no APT candidates.
if [[ -z "${LATEST_DRIVER}" && -z "${REQUESTED_DRIVER}" ]]; then
	if [[ -n "${FLAVOR_EXPLICIT}" ]]; then
		abort "No \`${EFFECTIVE_FLAVOR}\` driver candidates available from APT. Drop \`--${FLAVOR}\`, run \`sudo apt-get update\` to refresh the package cache, or pass \`--package=PKG\` to override."
	fi
	if [[ -n "${INSTALLED_DRIVER}" ]]; then
		# Never silently switch flavor on an installed system: falling back to the opposite
		# kernel-module flavor would be a root-level migration the user did not ask for.
		if [[ "${EFFECTIVE_FLAVOR}" == 'open' ]]; then
			opposite_flavor='proprietary'
		else
			opposite_flavor='open'
		fi
		abort "No \`${EFFECTIVE_FLAVOR}\` driver candidates available from APT to upgrade \`${INSTALLED_DRIVER}\`. Run \`sudo apt-get update\` to refresh the cache, pass \`--${opposite_flavor}\` to switch flavors explicitly, or pass \`--package=PKG\` to override."
	fi
	# Fresh-install fallback: pick whichever flavor the PPA publishes rather than refusing.
	warn "No \`${EFFECTIVE_FLAVOR}\` driver candidates available; falling back to \`${LATEST_FALLBACK}\` ($(driver-flavor "${LATEST_FALLBACK}") flavor)."
	LATEST_DRIVER="${LATEST_FALLBACK}"
fi
LATEST_DRIVER_VERSION="$(apt-candidate-version "${LATEST_DRIVER}")"

if [[ -z "${REQUESTED_DRIVER}" ]]; then
	if [[ -n "${LATEST}" || -z "${INSTALLED_DRIVER}" ]]; then
		REQUESTED_DRIVER="${LATEST_DRIVER}"
		REQUESTED_DRIVER_VERSION="${LATEST_DRIVER_VERSION}"
	elif [[ -n "${FLAVOR_EXPLICIT}" && "${INSTALLED_FLAVOR}" != "${FLAVOR}" ]]; then
		# Standalone `--open` / `--proprietary` (no `--latest`, no `--package`) on an
		# opposite-flavor install: switch to the same-major counterpart, e.g.
		# `--open` on `nvidia-driver-595` -> `nvidia-driver-595-open`.
		counterpart="$(toggle-open-suffix "${INSTALLED_DRIVER}")"
		if ! driver-available "${counterpart}"; then
			abort "Cannot switch \`${INSTALLED_DRIVER}\` to ${FLAVOR} flavor: \`${counterpart}\` is not available from APT. Pass \`--latest\` to switch to the latest ${FLAVOR} branch instead."
		fi
		REQUESTED_DRIVER="${counterpart}"
		REQUESTED_DRIVER_VERSION="$(apt-candidate-version "${REQUESTED_DRIVER}")"
	else
		REQUESTED_DRIVER="${INSTALLED_DRIVER}"
		REQUESTED_DRIVER_VERSION="${INSTALLED_DRIVER_CANDIDATE_VERSION}"
	fi
else
	REQUESTED_DRIVER_VERSION="$(apt-candidate-version "${REQUESTED_DRIVER}")"
	if [[ -z "${REQUESTED_DRIVER_VERSION}" ]]; then
		abort "Unable to locate package ${REQUESTED_DRIVER}."
	fi
	requested_flavor="$(driver-flavor "${REQUESTED_DRIVER}")"
	if [[ -n "${FLAVOR_EXPLICIT}" && "${requested_flavor}" != "${FLAVOR}" ]]; then
		suggested="$(toggle-open-suffix "${REQUESTED_DRIVER}")"
		conflict_msg="Option \`--${FLAVOR}\` conflicts with package \`${REQUESTED_DRIVER}\` (${requested_flavor} flavor)."
		if driver-available "${suggested}"; then
			conflict_msg+=" Did you mean \`${suggested}\`?"
		fi
		abort "${conflict_msg}"
	fi
fi

# `--upgrade-only` promises to keep the installed driver package, so reject any combination that
# would change the package name (including a flavor flag that rewrote `REQUESTED_DRIVER` to the
# counterpart, or an explicit `--package=` pointing at a different driver). Without this guard the
# later install path would purge the existing NVIDIA stack and reinstall the new package — a
# root-level system change masquerading as an upgrade.
if [[ -n "${UPGRADE_ONLY}" && -n "${INSTALLED_DRIVER}" && "${INSTALLED_DRIVER}" != "${REQUESTED_DRIVER}" ]]; then
	abort "Option \`--upgrade-only\` cannot change the driver package from \`${INSTALLED_DRIVER}\` to \`${REQUESTED_DRIVER}\`. Drop \`--upgrade-only\` to switch packages."
fi

ohai "Available NVIDIA drivers:"
for driver in "${AVAILABLE_DRIVERS[@]}"; do
	flavor="$(driver-flavor "${driver}")"
	prefix="    "
	if [[ "${driver}" == "${REQUESTED_DRIVER}" ]]; then
		prefix="--> "
	fi

	# Each flavor's branch tip gets a flavor-explicit annotation so "latest" cannot be confused
	# with the user's effective `--latest` target under flavor preservation.
	suffix=''
	if [[ "${driver}" == "${LATEST_PROPRIETARY_DRIVER}" ]]; then
		suffix=' (latest proprietary)'
	elif [[ "${driver}" == "${LATEST_OPEN_DRIVER}" ]]; then
		suffix=' (latest open)'
	fi

	# Dim other-flavor rows: picking one would require a purge+reinstall flavor switch. The
	# REQUESTED_DRIVER row is exempt so the `--> ` arrow keeps priority on a deliberate switch.
	dim=''
	if [[ -n "${INSTALLED_FLAVOR}" &&
		"${flavor}" != "${INSTALLED_FLAVOR}" &&
		"${driver}" != "${REQUESTED_DRIVER}" ]]; then
		dim=1
	fi

	if [[ "${driver}" == "${INSTALLED_DRIVER}" ]]; then
		if [[ "${driver}" != "${REQUESTED_DRIVER}" ]]; then
			prefix="<-- "
		elif [[ "${REQUESTED_DRIVER_VERSION}" != "${INSTALLED_DRIVER_VERSION}" ]]; then
			prefix="--> "
		else
			prefix="--- "
		fi
		if [[ "${INSTALLED_DRIVER_VERSION}" == "${INSTALLED_DRIVER_CANDIDATE_VERSION}" ]]; then
			if [[ "${driver}" == "${LATEST_DRIVER}" ]]; then
				echo "${prefix}${tty_green}${driver} [${INSTALLED_DRIVER_VERSION}]${tty_reset} ${tty_yellow}[installed]${tty_reset} (up-to-date)"
			else
				echo "${prefix}${tty_bold}${driver} [${INSTALLED_DRIVER_VERSION}]${tty_reset} ${tty_yellow}[installed]${tty_reset} (up-to-date)"
			fi
		else
			echo "${prefix}${tty_bold}${driver} [${INSTALLED_DRIVER_VERSION}]${tty_reset} ${tty_yellow}[installed]${tty_reset} (upgradable to [${INSTALLED_DRIVER_CANDIDATE_VERSION}])"
		fi
	elif [[ -n "${dim}" ]]; then
		echo "${prefix}${tty_dim}${driver} [$(apt-candidate-version "${driver}")]${suffix}${tty_reset}"
	elif [[ "${driver}" == "${LATEST_DRIVER}" ]]; then
		echo "${prefix}${tty_green}${driver} [${LATEST_DRIVER_VERSION}]${tty_reset}${suffix}"
	else
		echo "${prefix}${driver} [$(apt-candidate-version "${driver}")]${suffix}"
	fi
done

if [[ "${INSTALLED_DRIVER}@${INSTALLED_DRIVER_VERSION}" == "${REQUESTED_DRIVER}@${REQUESTED_DRIVER_VERSION}" ]]; then
	echo
	ohai "Your NVIDIA driver is already up-to-date."
	exit
elif [[ "${INSTALLED_DRIVER}" == "${REQUESTED_DRIVER}" && -z "${UPGRADE_ONLY}" && -z "${LATEST}" ]]; then
	echo
	ohai "The requested driver ${REQUESTED_DRIVER} is already installed. Run \`bash $(basename "$0") --upgrade-only\` to upgrade."
	exit
elif [[ -n "${DRY_RUN}" ]]; then
	exit
fi

echo

### Show the installation plan and wait for user confirmation ######################################

if [[ -z "${INSTALLED_DRIVER}" ]]; then
	ohai "Install the NVIDIA driver ${REQUESTED_DRIVER} [${REQUESTED_DRIVER_VERSION}]."
elif [[ "$(driver-major "${REQUESTED_DRIVER}")" -ge "$(driver-major "${INSTALLED_DRIVER}")" ]]; then
	ohai "Upgrade the NVIDIA driver from ${INSTALLED_DRIVER} [${INSTALLED_DRIVER_VERSION}] to ${REQUESTED_DRIVER} [${REQUESTED_DRIVER_VERSION}]."
else
	ohai "Downgrade the NVIDIA driver from ${INSTALLED_DRIVER} [${INSTALLED_DRIVER_VERSION}] to ${REQUESTED_DRIVER} [${REQUESTED_DRIVER_VERSION}]."
fi

REQUESTED_FLAVOR="$(driver-flavor "${REQUESTED_DRIVER}")"
if [[ -n "${INSTALLED_FLAVOR}" && "${INSTALLED_FLAVOR}" != "${REQUESTED_FLAVOR}" ]]; then
	ohai "Switching kernel-module flavor from ${INSTALLED_FLAVOR} to ${REQUESTED_FLAVOR}."
fi

# Preflight the install plan before any teardown so an unsatisfiable APT transaction aborts
# while the existing driver is still working. `--allow-change-held-packages` lets the simulate
# plan past the existing apt-mark holds (the real `apt-mark unhold` runs in the teardown block).
if [[ -n "${INSTALLED_DRIVER}" && "${INSTALLED_DRIVER}" != "${REQUESTED_DRIVER}" ]]; then
	exec_cmd "sudo apt-get install --simulate --allow-change-held-packages --yes ${REQUESTED_DRIVER}"
fi

DM_SERVICES=()
if [[ -n "$(sudo lsof -t /dev/nvidia* 2>/dev/null || true)" ]]; then
	for dm in gdm3 lightdm; do
		if service "${dm}" status &>/dev/null; then
			DM_SERVICES+=("${dm}")
		fi
	done
fi

if [[ "${#DM_SERVICES[@]}" -gt 0 ]]; then
	if [[ "${#DM_SERVICES[@]}" -gt 1 ]]; then
		warn "The following display manager services are running:"
	else
		warn "The following display manager service is running:"
	fi
	printf "  - %s\n" "${DM_SERVICES[@]}"
	echo "The service will be stopped during the installation which may shut down the GUI desktop. The service will be restarted after the installation."
fi

if [[ -z "${YES}" ]]; then
	ring_bell
	CONFIRM_MESSAGE="Press ${tty_bold}RETURN${tty_reset}/${tty_bold}ENTER${tty_reset} to continue or any other key to abort:" wait_for_user
fi

### Do install/upgrade the requested driver package ################################################

# Stop display manager services
for dm in "${DM_SERVICES[@]}"; do
	exec_cmd "sudo service ${dm} stop"
	# shellcheck disable=SC2064
	trap "exec_cmd 'sudo service ${dm} start'" EXIT # restart the service on exit
done

# Disable persistence mode
if [[ -n "$(sudo lsmod | grep '^nvidia' | awk '{ print $1 }')" ]]; then
	exec_cmd "sudo nvidia-smi -pm 0 || true"
fi

sleep 1 # ensure the processes are stopped

# Ensure no processes are using the NVIDIA devices
# shellcheck disable=SC2207
PIDS=($(sudo lsof -t /dev/nvidia* 2>/dev/null || true))
if [[ "${#PIDS[@]}" -gt 0 ]]; then
	cat >&2 <<EOS

${tty_red}FATAL:${tty_white} Some processes are still using the NVIDIA devices.${tty_reset}

$(ps --format=pid,user,command --pid="${PIDS[*]}")
EOS

	if [[ -z "${YES}" ]]; then
		ring_bell
		CONFIRM_MESSAGE="Press ${tty_bold}RETURN${tty_reset}/${tty_bold}ENTER${tty_reset} to kill (\`kill -9\`) and continue or any other key to abort:" wait_for_user
	fi

	exec_cmd "sudo kill -9 ${PIDS[*]}"
	sleep 3 # ensure the processes are stopped
fi

# Offload the NVIDIA kernel modules
MODULES="$(sudo lsmod | grep '^nvidia' | awk '{ print $1 }')"
MODULES="${MODULES//$'\n'/ }"
if [[ -n "${MODULES}" ]]; then
	exec_cmd "sudo modprobe -r -f ${MODULES}"
fi

if [[ -n "${INSTALLED_DRIVER}" ]]; then
	NVIDIA_PACKAGES="$(apt-list-nvidia-packages)"
	if [[ -n "${NVIDIA_PACKAGES}" ]]; then
		exec_cmd "sudo apt-mark unhold ${NVIDIA_PACKAGES}"
	fi
	if [[ "${INSTALLED_DRIVER}" == "${REQUESTED_DRIVER}" ]]; then
		# Upgrade the existing driver packages
		exec_cmd "sudo apt-get install --only-upgrade --yes ${NVIDIA_PACKAGES}"
	else
		# The install plan was already validated above with `apt-get install --simulate` (before
		# any destructive teardown), so this purge is safe to run.
		exec_cmd "sudo apt-get purge --yes ${NVIDIA_PACKAGES}"
	fi
fi

if [[ "${INSTALLED_DRIVER}" != "${REQUESTED_DRIVER}" ]]; then
	# Install the required driver packages
	exec_cmd "sudo apt-get install --yes ${REQUESTED_DRIVER}"
fi

# Hold the NVIDIA driver packages for `apt upgrade`
NVIDIA_PACKAGES="$(apt-list-nvidia-packages)"
if [[ -n "${NVIDIA_PACKAGES}" ]]; then
	exec_cmd "sudo apt-mark hold ${NVIDIA_PACKAGES}"
fi

### Reload the NVIDIA kernel modules ###############################################################

exec_cmd 'nvidia-smi'

# Enable persistence mode
exec_cmd "sudo nvidia-smi -pm 1 || true"
