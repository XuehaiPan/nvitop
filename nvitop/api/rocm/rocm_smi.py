# pylint: skip-file
# ruff: noqa
# flake8: noqa
# type: ignore
"""ROCm_SMI_LIB CLI Tool

====
Adapted by Junyi from 'rocm-smi-lib', branch `develop`, commit `9a3a50f`.
lint is disabled because this file is adapted from rocm-smi-lib.
https://github.com/ROCm/rocm_smi_lib/tree/develop/python_smi_tools
=====

This tool acts as a command line interface for manipulating
and monitoring the amdgpu kernel, and is intended to replace
and deprecate the existing rocm_smi.py CLI tool.
It uses Ctypes to call the rocm_smi_lib API.
Recommended: At least one AMD GPU with ROCm driver installed
Required: ROCm SMI library installed (librocm_smi64)
"""

import _thread
import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from subprocess import check_output
from time import ctime

from .rsmiBindings import *


# rocmSmiLib_cli version. Increment this as needed.
# Major version - Increment when backwards-compatibility breaks
# Minor version - Increment when adding a new feature, set to 0 when major is incremented
# Patch version - Increment when adding a fix, set to 0 when minor is incremented
# Hash  version - Shortened commit hash. Print here and not with lib for consistency with amd-smi
SMI_MAJ = 2
SMI_MIN = 0
SMI_PAT = 0
# SMI_HASH is provided by rsmiBindings
__version__ = f'{SMI_MAJ}.{SMI_MIN}.{SMI_PAT}+{SMI_HASH}'

# Set to 1 if an error occurs
RETCODE = 0

# If we want JSON format output instead
PRINT_JSON = False
JSON_DATA = {}
# Version of the JSON output used to save clocks
CLOCK_JSON_VERSION = 1

# Apply max buffer to all data allocation
MAX_BUFF_SIZE = 256

headerString = ' ROCm System Management Interface '
footerString = ' End of ROCm SMI Log '
# Output formatting
appWidth = 90
deviceList = []

# Enable or disable serialized format
OUTPUT_SERIALIZATION = False

# These are the valid clock types that can be returned/modified:
# TODO: "clk_type_names" from rsmiBindings.py should fetch valid clocks from
#       the same location as rocm_smi_device.cc instead of hardcoding the values
validClockNames = clk_type_names[1:-2]
# The purpose of the [1:-2] here ^^^^ is to remove the duplicate elements at the
# beginning and end of the clk_type_names list (specifically sclk and mclk)
# Also the "invalid" clock in the list is removed since it isn't a valid clock type
validClockNames.append('pcie')
validClockNames.sort()

rocmsmi = None


def initRsmiBindings(silent=False):
    """
    Modified by Junyi
    """
    path_librocm = ''

    def _find_lib_rocm():
        """search for librocm and returns path
        if search fails, returns empty string
        """
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        rocm_lib_path = os.path.join(rocm_path, 'lib/librocm_smi64.so')
        return rocm_lib_path if os.path.isfile(rocm_lib_path) else ''

    def print_silent(*args):
        if not silent:
            print(args)

    rocm_smi_lib_path = os.getenv('ROCM_SMI_LIB_PATH')
    if rocm_smi_lib_path != None:
        path_librocm = rocm_smi_lib_path
    else:
        path_librocm = _find_lib_rocm()

    try:
        cdll.LoadLibrary(path_librocm)
        return CDLL(path_librocm)
    except OSError:
        print(
            'Unable to load the rocm_smi library.\n'
            'Set LD_LIBRARY_PATH to the folder containing librocm_smi64.so.@VERSION_MAJOR@\n'
            '{}Please refer to https://github.com/'
            'RadeonOpenCompute/rocm_smi_lib for the installation guide.{}'.format(
                '\33[33m',
                '\033[0m',
            ),
        )
        exit()


def driverInitialized():
    """Returns true if amdgpu is found in the list of initialized modules"""
    driverInitialized = ''
    try:
        driverInitialized = str(
            subprocess.check_output('cat /sys/module/amdgpu/initstate |grep live', shell=True),
        )
    except subprocess.CalledProcessError:
        pass
    if len(driverInitialized) > 0:
        return True
    return False


def formatJson(device, log):
    """Print out in JSON format

    :param device: DRM device identifier
    :param log: String to parse and output into JSON format
    """
    global JSON_DATA
    for line in log.splitlines():
        # Drop any invalid or improperly-formatted data
        if ':' not in line:
            continue
        logTuple = line.split(': ')
        if str(device) != 'system':
            JSON_DATA['card' + str(device)][logTuple[0]] = logTuple[1].strip()
        else:
            JSON_DATA['system'][logTuple[0]] = logTuple[1].strip()


def formatCsv(deviceList):
    """Print out the JSON_DATA in CSV format"""
    global JSON_DATA
    jsondata = json.dumps(JSON_DATA)
    outstr = jsondata
    # Check if the first json data element is 'system' or 'device'
    outputType = outstr[outstr.find('"') + 1 :]
    outputType = outputType[: outputType.find('"')]
    header = []
    my_string = ''
    if outputType != 'system':
        header.append('device')
    else:
        header.append('system')
    if outputType == 'system':
        jsonobj = json.loads(jsondata)
        keylist = header
        for record in jsonobj['system']:
            my_string += '"{}", "{}"\n'.format(record, jsonobj['system'][record])
        # add header
        my_string = 'name, value\n' + my_string
        return my_string
    headerkeys = []
    # Separate device-specific information from system-level information
    for dev in deviceList:
        if str(dev) != 'system':
            headerkeys.extend(l for l in JSON_DATA['card' + str(dev)].keys() if l not in headerkeys)
        else:
            headerkeys.extend(l for l in JSON_DATA['system'].keys() if l not in headerkeys)
    header.extend(headerkeys)
    outStr = '%s\n' % ','.join(header)
    if len(header) <= 1:
        return ''
    for dev in deviceList:
        if str(dev) != 'system':
            outStr += 'card%s,' % dev
        else:
            outStr += 'system,'
        for val in headerkeys:
            try:
                if str(dev) != 'system':
                    # Remove commas like the ones in PCIe speed
                    outStr += '%s,' % JSON_DATA['card' + str(dev)][val].replace(',', '')
                else:
                    outStr += '%s,' % JSON_DATA['system'][val].replace(',', '')
            except KeyError:
                # If the key doesn't exist (like dcefclock on Fiji, or unsupported functionality)
                outStr += 'N/A,'
        # Drop the trailing ',' and replace it with a \n
        outStr = '%s\n' % outStr[0:-1]
    return outStr


def formatMatrixToJSON(deviceList, matrix, metricName):
    """ Format symmetric matrix of GPU permutations to become JSON print-ready.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param metricName: Title of the item to print to the log
    :param matrix: symmetric matrix full of values of every permutation of DRM devices.

    Matrix example:

    .. math::

        \\begin{bmatrix}
                 & GPU0 & GPU1 \\\\
            GPU0 & 0 & 40 \\\\
            GPU1 & 40 & 0
        \\end{bmatrix}

    Where matrix content is: [[0, 40], [40, 0]]
    """
    devices_ind = range(len(deviceList))
    for row_indx in devices_ind:
        # Start at row_indx +1 to avoid printing repeated values ( GPU1 x GPU2 is the same as GPU2 x GPU1 )
        for col_ind in range(row_indx + 1, len(deviceList)):
            try:
                valueStr = matrix[deviceList[row_indx]][deviceList[col_ind]].value
            except AttributeError:
                valueStr = matrix[deviceList[row_indx]][deviceList[col_ind]]

            printSysLog(metricName.format(deviceList[row_indx], deviceList[col_ind]), valueStr)


def getBus(device, silent=False):
    """Return the bus identifier of a given device

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    bdfid = c_uint64(0)
    ret = rocmsmi.rsmi_dev_pci_id_get(device, byref(bdfid))

    # BDFID = ((DOMAIN & 0xFFFFFFFF) << 32) | ((PARTITION_ID & 0xF) << 28) | ((BUS & 0xFF) << 8) |
    # ((DEVICE & 0x1F) <<3 ) | (FUNCTION & 0x7)
    # bits [63:32] = domain
    # bits [31:28] = partition id
    # bits [27:16] = reserved
    # bits [15: 0] = pci bus/device/function
    domain = (bdfid.value >> 32) & 0xFFFFFFFF
    bus = (bdfid.value >> 8) & 0xFF
    device = (bdfid.value >> 3) & 0x1F
    function = bdfid.value & 0x7

    pic_id = f'{domain:04X}:{bus:02X}:{device:02X}.{function:0X}'
    if rsmi_ret_ok(ret, device, 'get_pci_id', silent):
        return pic_id


def getPartitionId(device, silent=False):
    """Return the partition identifier of a given device

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    bdfid = c_uint64(0)
    ret = rocmsmi.rsmi_dev_pci_id_get(device, byref(bdfid))

    # BDFID = ((DOMAIN & 0xFFFFFFFF) << 32) | ((PARTITION_ID & 0xF) << 28) | ((BUS & 0xFF) << 8) |
    # ((DEVICE & 0x1F) <<3 ) | (FUNCTION & 0x7)
    # bits [63:32] = domain
    # bits [31:28] = partition id
    # bits [27:16]  = reserved
    # bits [15: 0]  = pci bus/device/function
    partition_num = (bdfid.value >> 28) & 0xF
    pci_id = bdfid.value
    partition_id = f'{partition_num:d}'
    if rsmi_ret_ok(ret, device, 'get_pci_id', silent):
        return partition_id


def getFanSpeed(device, silent=True):
    """Return a tuple with the fan speed (value,%) for a specified device,
    or (None,None) if either current fan speed or max fan speed cannot be
    obtained

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is on.
    """
    fanLevel = c_int64()
    fanMax = c_int64()
    sensor_ind = c_uint32(0)
    fl = 0
    fm = 0

    """ If ret = 2; (No such file or directory)
    /sys/class/drm/cardX/device/hwmon/hwmonX/pwmX
    """
    ret = rocmsmi.rsmi_dev_fan_speed_get(device, sensor_ind, byref(fanLevel))
    if rsmi_ret_ok(ret, device, 'get_fan_speed', silent):
        fl = fanLevel.value
    last_ret = ret

    """ If ret = 2; (No such file or directory)
    /sys/class/drm/cardX/device/hwmon/hwmonX/pwmX
    """
    ret = rocmsmi.rsmi_dev_fan_speed_max_get(device, sensor_ind, byref(fanMax))
    if rsmi_ret_ok(ret, device, 'get_fan_max_speed', silent):
        fm = fanMax.value

    """ In case we had an error before, we don't overwrite it with a
        possible success now. Otherwise, we get the next error.
    """
    if last_ret == rsmi_status_t.RSMI_STATUS_SUCCESS:
        last_ret = ret

    if fl == 0 or fm == 0:
        return (last_ret, fl, 0)  # to prevent division by zero crash

    return (last_ret, fl, round((float(fl) / float(fm)) * 100, 2))


def getGpuUse(device, silent=False):
    """Return the current GPU usage as a percentage

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    percent = c_uint32()
    ret = rocmsmi.rsmi_dev_busy_percent_get(device, byref(percent))
    if rsmi_ret_ok(ret, device, 'GPU Utilization ', silent):
        return percent.value
    return -1


def getDRMDeviceId(device, silent=False):
    """Return the hexadecimal value of a device's ID

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    dv_id = c_short()
    ret = rocmsmi.rsmi_dev_id_get(device, byref(dv_id))
    device_id_ret = 'N/A'
    if rsmi_ret_ok(ret, device, 'get_device_id', silent):
        device_id_ret = hex(dv_id.value)
    return device_id_ret


def getRev(device, silent=False):
    """Return the hexadecimal value of a device's Revision

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    dv_rev = c_short()
    ret = rocmsmi.rsmi_dev_revision_get(device, byref(dv_rev))
    revision_ret = 'N/A'
    if rsmi_ret_ok(ret, device, 'get_device_rev', silent=silent):
        revision_ret = padHexValue(hex(dv_rev.value), 2)
    return revision_ret


def getSubsystemId(device, silent=False):
    """Return the a device's subsystem id

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    model = create_string_buffer(MAX_BUFF_SIZE)
    ret = rocmsmi.rsmi_dev_subsystem_name_get(device, model, MAX_BUFF_SIZE)
    device_model = 'N/A'
    if rsmi_ret_ok(ret, device, 'get_subsystem_name', silent=silent):
        device_model = model.value.decode()
        # padHexValue is used for applications that expect 4-digit card models
        device_model = padHexValue(device_model, 4)
    return device_model


def getVendor(device, silent=False):
    """Return the a device's vendor id

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    vendor = create_string_buffer(MAX_BUFF_SIZE)
    device_vendor = 'N/A'
    # Retrieve card vendor
    ret = rocmsmi.rsmi_dev_vendor_name_get(device, vendor, MAX_BUFF_SIZE)
    # Only continue if GPU vendor is AMD
    if rsmi_ret_ok(ret, device, 'get_vendor_name', silent) and isAmdDevice(device):
        device_vendor = vendor.value.decode()
    return device_vendor


def getGUID(device, silent=False):
    """Return the uint64 value of device's GUID,
    also referred as GPU ID - reported by KFD.

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    guid = c_uint64()
    ret = rocmsmi.rsmi_dev_guid_get(device, byref(guid))
    guid_ret = 'N/A'
    if rsmi_ret_ok(ret, device, 'get_gpu_id_kfd', silent=silent):
        guid_ret = guid.value
    return guid_ret


def getTargetGfxVersion(device, silent=False):
    """Return the uint64 value of device's target
    graphics version as reported by KFD

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    gfx_version = c_uint64()
    gfx_ver_ret = 'N/A'
    ret = rocmsmi.rsmi_dev_target_graphics_version_get(device, byref(gfx_version))
    if rsmi_ret_ok(ret, device, 'get_target_gfx_version', silent=silent):
        gfx_ver_ret = 'gfx' + str(gfx_version.value)
    return gfx_ver_ret


def getNodeId(device, silent=False):
    """Return the uint32 value of device's node id
    reported by KFD.

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    node_id = c_uint32()
    ret = rocmsmi.rsmi_dev_node_id_get(device, byref(node_id))
    node_id_ret = 'N/A'
    if rsmi_ret_ok(ret, device, 'get_node_id_kfd', silent=silent):
        node_id_ret = node_id.value
    return node_id_ret


def getDeviceName(device, silent=False):
    """Return the uint64 value of device's target
        graphics version as reported by KFD

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    # Retrieve the device series
    series = create_string_buffer(MAX_BUFF_SIZE)
    device_name_ret = 'N/A'
    ret = rocmsmi.rsmi_dev_name_get(device, series, MAX_BUFF_SIZE)
    if rsmi_ret_ok(ret, device, 'get_name', silent=silent):
        device_name_ret = series.value.decode()
    return device_name_ret


def getMaxPower(device, silent=False):
    """Return the maximum power cap of a given device

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    power_cap = c_uint64()
    ret = rocmsmi.rsmi_dev_power_cap_get(device, 0, byref(power_cap))
    if rsmi_ret_ok(ret, device, 'get_power_cap', silent):
        return power_cap.value / 1000000
    return -1


def getMemInfo(device, memType, silent=False):
    """Returns a tuple of (memory_used, memory_total) of
        the requested memory type usage for the device specified

    :param device: DRM device identifier
    :param type: [vram|vis_vram|gtt] Memory type to return
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off,
        which exposes any issue accessing the different
        memory types.
    """
    memType = memType.upper()
    if memType not in memory_type_l:
        printErrLog(device, 'Invalid memory type %s' % (memType))
        return (None, None)

    memoryUse = c_uint64()
    memoryTot = c_uint64()
    memUsed = None
    memTotal = None

    ret = rocmsmi.rsmi_dev_memory_usage_get(device, memory_type_l.index(memType), byref(memoryUse))
    if rsmi_ret_ok(ret, device, 'get_memory_usage_' + str(memType), silent):
        memUsed = memoryUse.value

    ret = rocmsmi.rsmi_dev_memory_total_get(device, memory_type_l.index(memType), byref(memoryTot))
    if rsmi_ret_ok(ret, device, 'get_memory_total_' + str(memType), silent):
        memTotal = memoryTot.value
    return (memUsed, memTotal)


def getProcessName(pid):
    """Get the process name of a specific pid

    :param pid: Process ID of a program to be parsed
    """
    if int(pid) < 1:
        logging.debug('PID must be greater than 0')
        return 'UNKNOWN'
    try:
        pName = str(subprocess.check_output('ps -p %d -o comm=' % (int(pid)), shell=True))
    except subprocess.CalledProcessError:
        pName = 'UNKNOWN'

    if pName == None:
        pName = 'UNKNOWN'

    # Remove the substrings surrounding from process name (b' and \n')
    if str(pName).startswith("b'"):
        pName = pName[2:]
    if str(pName).endswith("\\n'"):
        pName = pName[:-3]

    return pName


def getPerfLevel(device, silent=False):
    """Return the current performance level of a given device

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    perf = rsmi_dev_perf_level_t()
    ret = rocmsmi.rsmi_dev_perf_level_get(device, byref(perf))
    if rsmi_ret_ok(ret, device, 'get_perf_level', silent):
        return perf_level_string(perf.value)
    return 'N/A'


def getPid(name):
    """Get the process id of a specific application

    :param name: Process name of a program to be parsed
    """
    return check_output(['pidof', name])


def getPidList():
    """Return a list of KFD process IDs"""
    num_items = c_uint32()
    ret = rocmsmi.rsmi_compute_process_info_get(None, byref(num_items))
    if rsmi_ret_ok(ret, metric='get_compute_process_info'):
        buff_sz = num_items.value + 10
        procs = (rsmi_process_info_t * buff_sz)()
        procList = []
        ret = rocmsmi.rsmi_compute_process_info_get(byref(procs), byref(num_items))
        for i in range(num_items.value):
            procList.append('%s' % (procs[i].process_id))
        return procList
    return None


def getPower(device):
    """Return dictionary of power responses.
        Response power dictionary:

        .. code-block:: python

            {
                'power': string wattage response or 'N/A' (for not RSMI_STATUS_SUCCESS),
                'power_type': power type string - 'Current Socket' or 'Average',
                'unit': W (Watt)
                'ret': response of rsmi_dev_power_get(device, byref(power), byref(power_type))
            }

    :param device: DRM device identifier
    """

    power = c_int64(0)
    power_type = rsmi_power_type_t()
    power_ret_dict = {
        'power': 'N/A',
        'power_type': 'N/A',
        'unit': 'W',
        'ret': rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED,
    }
    ret = rocmsmi.rsmi_dev_power_get(device, byref(power), byref(power_type))
    if ret == rsmi_status_t.RSMI_STATUS_SUCCESS:
        power_ret_dict = {
            'power': str(power.value / 1000000),
            'power_type': rsmi_power_type_dict[power_type.value],
            'unit': 'W',
            'ret': ret,
        }
    else:
        power_ret_dict['ret'] = ret
    return power_ret_dict


def getRasEnablement(device, block, silent=True):
    """Return RAS enablement state for a given device

    :param device: DRM device identifier
    :param block: RAS block identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is on.
    """
    state = rsmi_ras_err_state_t()
    ret = rocmsmi.rsmi_dev_ecc_status_get(device, rsmi_gpu_block_d[block], byref(state))

    if rsmi_ret_ok(ret, device, 'get_ecc_status_' + str(block), silent):
        return rsmi_ras_err_stale_machine[state.value].upper()
    return 'N/A'


def getTemp(device, sensor, silent=True):
    """Display the current temperature from a given device's sensor

    :param device: DRM device identifier
    :param sensor: Temperature sensor identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is on.
    """
    temp = c_int64(0)
    metric = rsmi_temperature_metric_t.RSMI_TEMP_CURRENT
    ret = rocmsmi.rsmi_dev_temp_metric_get(
        c_uint32(device),
        temp_type_lst.index(sensor),
        metric,
        byref(temp),
    )
    if rsmi_ret_ok(ret, device, 'get_temp_metric' + str(sensor), silent):
        return temp.value / 1000
    return 'N/A'


def findFirstAvailableTemp(device):
    """Discovers the first available device temperature to display

        Returns a tuple of (temp_type, temp_value) for the device specified

    :param device: DRM device identifier
    """
    temp = c_int64(0)
    metric = rsmi_temperature_metric_t.RSMI_TEMP_CURRENT
    ret_temp = 'N/A'
    ret_temp_type = temp_type_lst[0]
    for i, templist_val in enumerate(temp_type_lst):
        ret = rocmsmi.rsmi_dev_temp_metric_get(c_uint32(device), i, metric, byref(temp))
        if rsmi_ret_ok(ret, device, 'get_temp_metric_' + templist_val, silent=True):
            ret_temp = temp.value / 1000
            ret_temp_type = '(' + templist_val.capitalize() + ')'
            break
        else:
            continue
    return (ret_temp_type, ret_temp)


def getTemperatureLabel(deviceList):
    """Discovers the the first identified power label
        Returns a string label value

    :param device: DRM device identifier
    """
    # Default label is Edge
    tempLabel = temp_type_lst[0].lower()
    if len(deviceList) < 1:
        return tempLabel
    (temp_type, _) = findFirstAvailableTemp(deviceList[0])
    tempLabel = temp_type.lower().replace('(', '').replace(')', '')
    return tempLabel


def getPowerLabel(deviceList):
    """Discovers the the first identified power label

        Returns a string label value

    :param device: DRM device identifier
    """
    power = c_int64(0)
    # Default label is AvgPower
    powerLabel = rsmi_power_label.AVG_POWER
    if len(deviceList) < 1:
        return powerLabel
    device = deviceList[0]
    power_dict = getPower(device)
    if (
        power_dict['ret'] == rsmi_status_t.RSMI_STATUS_SUCCESS
        and power_dict['power_type'] == 'CURRENT SOCKET'
    ):
        powerLabel = rsmi_power_label.CURRENT_SOCKET_POWER
    return powerLabel


def getVbiosVersion(device, silent=False):
    """Returns the VBIOS version for a given device

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    vbios = create_string_buffer(256)
    ret = rocmsmi.rsmi_dev_vbios_version_get(device, vbios, 256)
    vbios_ret = 'N/A'
    if rsmi_ret_ok(ret, device, silent=silent):
        vbios_ret = vbios.value.decode()
        if vbios_ret == '':
            vbios_ret = 'N/A'
    return vbios_ret


def getVersion(deviceList, component, silent=False):
    """Return the software version for the specified component

    :param deviceList: List of DRM devices (can be a single-item list)
    :param component: Component (currently only driver)
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is off.
    """
    ver_str = create_string_buffer(256)
    ret = rocmsmi.rsmi_version_str_get(component, ver_str, 256)
    if rsmi_ret_ok(ret, None, 'get_version_str_' + str(component), silent):
        return ver_str.value.decode()
    return None


def getComputePartition(device, silent=True):
    """Return the current compute partition of a given device

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is on.
    """
    currentComputePartition = create_string_buffer(MAX_BUFF_SIZE)
    ret = rocmsmi.rsmi_dev_compute_partition_get(device, currentComputePartition, MAX_BUFF_SIZE)
    if (
        rsmi_ret_ok(ret, device, 'get_compute_partition', silent)
        and currentComputePartition.value.decode()
    ):
        return str(currentComputePartition.value.decode())
    return 'N/A'


def getMemoryPartition(device, silent=True):
    """Return the current memory partition of a given device

    :param device: DRM device identifier
    :param silent: Turn on to silence error output
        (you plan to handle manually). Default is on.
    """
    currentMemoryPartition = create_string_buffer(MAX_BUFF_SIZE)
    ret = rocmsmi.rsmi_dev_memory_partition_get(device, currentMemoryPartition, MAX_BUFF_SIZE)
    if (
        rsmi_ret_ok(ret, device, 'get_memory_partition', silent)
        and currentMemoryPartition.value.decode()
    ):
        return str(currentMemoryPartition.value.decode())
    return 'N/A'


def print2DArray(dataArray):
    """Print 2D Array with uniform spacing"""
    global PRINT_JSON
    dataArrayLength = []
    isPid = False
    if str(dataArray[0][0]) == 'PID':
        isPid = True
    for position in range(len(dataArray[0])):
        dataArrayLength.append(len(dataArray[0][position]))
    for position in range(len(dataArray)):
        for cell in range(len(dataArray[0])):
            if len(dataArray[position][cell]) > dataArrayLength[cell]:
                dataArrayLength[cell] = len(dataArray[position][cell])
    for position in range(len(dataArray)):
        printString = ''
        for cell in range(len(dataArray[0])):
            printString += str(dataArray[position][cell]).ljust(dataArrayLength[cell], ' ') + '\t'
        if PRINT_JSON:
            printString = ' '.join(printString.split()).lower()
            firstElement = printString.split(' ', 1)[0]
            printString = printString.split(' ', 1)[1]
            printString = printString.replace(' ', ', ')
            if position > 0:
                if isPid:
                    printSysLog('PID%s' % (firstElement), printString)
                else:
                    printSysLog(firstElement, printString)
        else:
            printLog(None, printString, None)


def printEmptyLine():
    """Print out a single empty line"""
    global PRINT_JSON
    if not PRINT_JSON:
        print()


def printErrLog(device, err):
    """Print out an error to the SMI log

    :param device: DRM device identifier
    :param err: Error string to print
    """
    global PRINT_JSON
    devName = device
    for line in err.split('\n'):
        errstr = f'GPU[{devName}]\t: {line}'
        if not PRINT_JSON:
            logging.error(errstr)
        else:
            logging.debug(errstr)


def printInfoLog(device, metricName, value):
    """Print out an info line to the SMI log

    :param device: DRM device identifier
    :param metricName: Title of the item to print to the log
    :param value: The item's value to print to the log
    """
    global PRINT_JSON

    if not PRINT_JSON:
        if value is not None:
            logstr = f'GPU[{device}]\t: {metricName}: {value}'
        else:
            logstr = f'GPU[{device}]\t: {metricName}'
        if device is None:
            logstr = logstr[13:]

        logging.info(logstr)


def printEventList(device, delay, eventList):
    """Print out notification events for a specified device

    :param device: DRM device identifier
    :param delay: Notification delay in ms
    :param eventList: List of event type names (can be a single-item list)
    """
    mask = 0
    ret = rocmsmi.rsmi_event_notification_init(device)
    if not rsmi_ret_ok(ret, device, 'event_notification_init'):
        printErrLog(device, 'Unable to initialize event notifications.')
        return
    for eventType in eventList:
        mask |= 2 ** notification_type_names.index(eventType.upper())
    ret = rocmsmi.rsmi_event_notification_mask_set(device, mask)
    if not rsmi_ret_ok(ret, device, 'set_event_notification_mask'):
        printErrLog(device, 'Unable to set event notification mask.')
        return
    while 1:  # Exit condition from user keyboard input of 'q' or 'ctrl + c'
        num_elements = c_uint32(1)
        data = rsmi_evt_notification_data_t(1)
        rocmsmi.rsmi_event_notification_get(delay, byref(num_elements), byref(data))
        if len(data.message) > 0:
            print2DArray(
                [
                    [
                        '\rGPU[%d]:\t' % (data.dv_ind),
                        ctime().split()[3],
                        notification_type_names[data.event.value - 1],
                        data.message.decode('utf8') + '\r',
                    ],
                ],
            )


def printLog(device, metricName, value=None, extraSpace=False, useItalics=False):
    """Print out to the SMI log

    :param device: DRM device identifier
    :param metricName: Title of the item to print to the log
    :param value: The item's value to print to the log
    """
    red = '\033[91m'
    green = '\033[92m'
    blue = '\033[94m'
    bold = '\033[1m'
    italics = '\033[3m'
    underline = '\033[4m'
    end = '\033[0m'
    global PRINT_JSON
    if PRINT_JSON:
        if value is not None and device is not None:
            formatJson(device, str(metricName) + ': ' + str(value))
        elif device is not None:
            formatJson(device, str(metricName))
        return
    if value is not None:
        logstr = f'GPU[{device}]\t\t: {metricName}: {value}'
    else:
        logstr = f'GPU[{device}]\t\t: {metricName}'
    if device is None:
        logstr = logstr.split(':', 1)[1][1:]
    # Force thread safe printing
    lock = multiprocessing.Lock()
    lock.acquire()
    if useItalics:
        logstr = italics + logstr + end
    try:
        if extraSpace:
            print('\n', end='')
        print(logstr + '\n', end='')
        sys.stdout.flush()
    # when piped into programs like 'head' - print throws an error.
    # silently ignore instead
    except (OSError, BrokenPipeError):
        # https://docs.python.org/3/library/signal.html#note-on-sigpipe
        # Python flushes standard streams on exit; redirect remaining output
        # to devnull to avoid another BrokenPipeError at shutdown
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)  # Python exits with error code 1 on EPIPE

    lock.release()


def printListLog(metricName, valuesList):
    """Print out to the SMI log for the lists

    :param metricName: Title of the item to print to the log
    :param valuesList: The item's list of values to print to the log
    """
    global PRINT_JSON
    listStr = ''
    line = metricName + ':\n'
    if not valuesList:
        line = 'None'
    else:
        for value in valuesList:
            value = str(value) + ' '
            if (len(line) + len(value)) < appWidth:
                line += value
            else:
                listStr = listStr + line + '\n'
                line = value
    if not PRINT_JSON:
        print(listStr + line)


def printLogSpacer(displayString=None, fill='=', contentSizeToFit=0):
    """Prints [name of the option]/[name of the program] in the spacer to explain data below

        If no parameters are given, a default fill of the '=' string is used in the spacer

    :param displayString: name of item to be displayed inside of the log spacer
    :param fill: padding string which surrounds the given display string
    :param contentSizeToFit: providing an integer > 0 allows
        ability to dynamically change output padding/fill based on this value
        instead of appWidth. Handy for concise info output.
    """
    global appWidth, PRINT_JSON
    resizeValue = appWidth
    if contentSizeToFit != 0:
        resizeValue = contentSizeToFit
    if resizeValue % 2:  # if odd -> make even
        resizeValue += 1
    # leaving below to check if resizing works properly
    # print("resizeVal=" +str(resizeValue) + "; appWidth=" + str(appWidth) +
    #       "; contentSizeToFit=" + str(contentSizeToFit) + "; fill=" + fill)

    if not PRINT_JSON:
        if displayString:
            if len(displayString) % 2:
                displayString += fill
            logSpacer = (
                fill * int((resizeValue - (len(displayString))) / 2)
                + displayString
                + fill * int((resizeValue - (len(displayString))) / 2)
            )
        else:
            logSpacer = fill * resizeValue
        print(logSpacer)


def printSysLog(SysComponentName, value):
    """Print out to the SMI log for repeated features

    :param SysComponentName: Title of the item to print to the log
    :param value: The item's value to print to the log
    """
    global PRINT_JSON, JSON_DATA
    if PRINT_JSON:
        if 'system' not in JSON_DATA:
            JSON_DATA['system'] = {}
        formatJson('system', str(SysComponentName) + ': ' + str(value))
        return

    logstr = f'{SysComponentName}: {value}'
    logging.debug(logstr)
    print(logstr)


def printTableLog(
    column_headers,
    data_matrix,
    device=None,
    tableName=None,
    anchor='>',
    v_delim='  ',
):
    """Print out to the SMI log for the lists

    :param column_headers: Header names for each column
    :param data_matrix: Matrix of values
    :param device: DRM device identifier
    :param tableName: Title of the table to print to the log
    :param anchor: Alignment direction of the print output
    :param v_delim: Boundary String delimiter for the print output
    """
    # Usage: the length of col_Names would be determining column width.
    # If additional space is needed, please pad corresponding column name with spaces
    # If table should print tabulated, pad name of column one with leading zeroes
    # Use anchor '<' to to align columns to the right
    global OUTPUT_SERIALIZATION, PRINT_JSON
    if OUTPUT_SERIALIZATION or PRINT_JSON:
        return

    if (device is not None) or tableName:
        if device is not None:
            print('\nGPU[%s]: ' % (device), end='\t')
        if tableName:
            print(tableName, end='')
        printEmptyLine()

    for header in column_headers:
        print(f'{header:>}', end=v_delim)
    printEmptyLine()

    for row in data_matrix:
        for index, cell in enumerate(row):
            if cell is None:
                cell = 'None'
            print(
                '{:{anc}{width}}'.format(cell, anc=anchor, width=len(column_headers[index])),
                end=v_delim,
            )
        printEmptyLine()


def printTableRow(space, displayString, v_delim=' '):
    """Print out a line of a matrix table

    :param space: The item's spacing to print
    :param displayString: The item's value to print
    :param v_delim: Boundary String delimiter for the print output
    """
    if space:
        print(space % (displayString), end=v_delim)
    else:
        print(displayString, end=v_delim)


def checkIfSecondaryDie(device):
    """Checks if GCD(die) is the secondary die in a MCM.
    MI200 device specific feature check.
    The secondary dies lacks power management features.

    :param device: The device to check
    """
    energy_count = c_uint64()
    counter_resoution = c_float()
    timestamp = c_uint64()

    # secondary die can be determined by checking if energy counter == 0
    ret = rocmsmi.rsmi_dev_energy_count_get(
        device,
        byref(energy_count),
        byref(counter_resoution),
        byref(timestamp),
    )
    if (rsmi_ret_ok(ret, None, 'energy_count_secondary_die_check', silent=False)) and (
        energy_count.value == 0
    ):
        return True
    return False


def resetClocks(deviceList):
    """Reset clocks to default

    Reset clocks to default values by setting performance level to auto, as well
    as setting OverDrive back to 0

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Reset Clocks ')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_overdrive_level_set(device, rsmi_dev_perf_level_t(0))
        if rsmi_ret_ok(ret, device, 'set_overdrive_level'):
            printLog(device, 'OverDrive set to 0', None)
        else:
            printLog(device, 'Unable to reset OverDrive', None)
        ret = rocmsmi.rsmi_dev_perf_level_set(device, rsmi_dev_perf_level_t(0))
        if rsmi_ret_ok(ret, device, 'set_perf_level'):
            printLog(device, 'Successfully reset clocks', None)
        else:
            printLog(device, 'Unable to reset clocks', None)
        ret = rocmsmi.rsmi_dev_perf_level_set(device, rsmi_dev_perf_level_t(0))
        if rsmi_ret_ok(ret, device, 'set_perf_level'):
            printLog(device, 'Performance level reset to auto', None)
        else:
            printLog(device, 'Unable to reset performance level to auto', None)


def resetFans(deviceList):
    """Reset fans to driver control for a list of devices.

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Reset GPU Fan Speed ')
    for device in deviceList:
        sensor_ind = c_uint32(0)
        ret = rocmsmi.rsmi_dev_fan_reset(device, sensor_ind)
        if rsmi_ret_ok(ret, device, silent=True):
            printLog(device, 'Successfully reset fan speed to driver control', None)
        else:
            printLog(device, 'Not supported on the given system', None)
    printLogSpacer()


def resetPowerOverDrive(deviceList, autoRespond):
    """Reset Power OverDrive to the default value

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    setPowerOverDrive(deviceList, 0, autoRespond)


def resetProfile(deviceList):
    """Reset profile for a list of a devices.

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Reset Profile ')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_power_profile_set(device, 0, profileString('BOOTUP DEFAULT'))
        if rsmi_ret_ok(ret, device, 'set_power_profile'):
            printLog(device, 'Successfully reset Power Profile', None)
        else:
            printErrLog(device, 'Unable to reset Power Profile')
        ret = rocmsmi.rsmi_dev_perf_level_set(device, rsmi_dev_perf_level_t(0))
        if rsmi_ret_ok(ret, device, 'set_perf_level'):
            printLog(device, 'Successfully reset Performance Level', None)
        else:
            printErrLog(device, 'Unable to reset Performance Level')
    printLogSpacer()


def resetXgmiErr(deviceList):
    """Reset the XGMI Error value

    :param deviceList: Reset XGMI error count for these devices
    """
    printLogSpacer('Reset XGMI Error Status ')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_xgmi_error_reset(device)
        if rsmi_ret_ok(ret, device, 'reset xgmi'):
            printLog(device, 'Successfully reset XGMI Error count', None)
        else:
            logging.error('GPU[%s]\t\t: Unable to reset XGMI error count', device)
    printLogSpacer()


def resetPerfDeterminism(deviceList):
    """Reset Performance Determinism

    :param deviceList: Disable Performance Determinism for these devices
    """
    printLogSpacer('Disable Performance Determinism')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_perf_level_set(device, rsmi_dev_perf_level_t(0))
        if rsmi_ret_ok(ret, device, 'disable performance determinism'):
            printLog(device, 'Successfully disabled performance determinism', None)
        else:
            logging.error('GPU[%s]\t\t: Unable to disable performance determinism', device)
    printLogSpacer()


def resetComputePartition(deviceList):
    """Reset Compute Partition to its boot state

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Reset compute partition to its boot state ')
    for device in deviceList:
        originalPartition = getComputePartition(device)
        ret = rocmsmi.rsmi_dev_compute_partition_reset(device)
        if rsmi_ret_ok(ret, device, 'reset_compute_partition', silent=True):
            resetBootState = getComputePartition(device)
            printLog(
                device,
                'Successfully reset compute partition ('
                + originalPartition
                + ') to boot state ('
                + resetBootState
                + ')',
                None,
            )
        elif ret == rsmi_status_t.RSMI_STATUS_PERMISSION:
            printLog(device, 'Permission denied', None)
        elif ret == rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED:
            printLog(device, 'Not supported on the given system', None)
        elif ret == rsmi_status_t.RSMI_STATUS_BUSY:
            printLog(device, 'Device is currently busy, try again later', None)
        else:
            rsmi_ret_ok(ret, device, 'reset_compute_partition')
            printErrLog(device, 'Failed to reset the compute partition to boot state')
    printLogSpacer()


def resetMemoryPartition(deviceList):
    """Reset current memory partition to its boot state

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Reset memory partition to its boot state ')
    for device in deviceList:
        originalPartition = getMemoryPartition(device)
        t1 = multiprocessing.Process(
            target=showProgressbar,
            args=('Resetting memory partition', 13),
        )
        t1.start()
        addExtraLine = True
        start = time.time()
        ret = rocmsmi.rsmi_dev_memory_partition_reset(device)
        stop = time.time()
        duration = stop - start
        if t1.is_alive():
            t1.terminate()
            t1.join()
        if duration < 0.1:  # For longer runs, add extra line before output
            addExtraLine = False  # This is to prevent overriding progress bar
        if rsmi_ret_ok(ret, device, 'reset_memory_partition', silent=True):
            resetBootState = getMemoryPartition(device)
            printLog(
                device,
                'Successfully reset memory partition ('
                + originalPartition
                + ') to boot state ('
                + resetBootState
                + ')',
                None,
                addExtraLine,
            )
        elif ret == rsmi_status_t.RSMI_STATUS_PERMISSION:
            printLog(device, 'Permission denied', None, addExtraLine)
        elif ret == rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED:
            printLog(device, 'Not supported on the given system', None, addExtraLine)
        elif ret == rsmi_status_t.RSMI_STATUS_BUSY:
            printLog(device, 'Device is currently busy, try again later', None)
        else:
            rsmi_ret_ok(ret, device, 'reset_memory_partition')
            printErrLog(device, 'Failed to reset memory partition to boot state')
    printLogSpacer()


def setClockRange(deviceList, clkType, minvalue, maxvalue, autoRespond):
    """Set the range for the specified clktype in the PowerPlay table for a list of devices.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param clktype: [sclk|mclk] Which clock type to apply the range to
    :param minvalue: Minimum value to apply to the clock range
    :param maxvalue: Maximum value to apply to the clock range
    :param autoRespond: Response to automatically provide for all prompts
    """
    global RETCODE
    if clkType not in {'sclk', 'mclk'}:
        printLog(None, 'Invalid range identifier %s' % (clkType), None)
        logging.error('Unsupported range type %s', clkType)
        RETCODE = 1
        return
    try:
        int(minvalue) & int(maxvalue)
    except ValueError:
        printErrLog(None, 'Unable to set %s range' % (clkType))
        logging.exception('%s or %s is not an integer', minvalue, maxvalue)
        RETCODE = 1
        return
    confirmOutOfSpecWarning(autoRespond)
    printLogSpacer(' Set Valid %s Range ' % (clkType))
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_clk_range_set(
            device,
            int(minvalue),
            int(maxvalue),
            rsmi_clk_names_dict[clkType],
        )
        if rsmi_ret_ok(ret, device, silent=True):
            printLog(
                device,
                f'Successfully set {clkType} from {minvalue}(MHz) to {maxvalue}(MHz)',
                None,
            )
        else:
            printErrLog(
                device,
                f'Unable to set {clkType} from {minvalue}(MHz) to {maxvalue}(MHz)',
            )
            RETCODE = 1
            if ret == rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED:
                printLog(
                    device,
                    'Setting %s range is not supported for this device.' % (clkType),
                    None,
                )


def setClockExtremum(deviceList, level, clkType, clkValue, autoRespond):
    """Set the range for the specified clktype in the PowerPlay table for a list of devices.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param level: [min|max] Minimum value or Maximum value
    :param clktype: [sclk|mclk] Which clock type to apply the range to
    :param clkValue: clock value to apply to the level
    :param autoRespond: Response to automatically provide for all prompts
    """
    global RETCODE
    if level not in {'min', 'max'}:
        printLog(None, 'Invalid extremum identifier %s, use min or max' % (level), None)
        logging.error('Unsupported clock extremum %s', level)
        RETCODE = 1
        return

    if clkType not in {'sclk', 'mclk'}:
        printLog(None, 'Invalid clock type identifier %s, use sclk or mclk ' % (clkType), None)
        logging.error('Unsupported clock type %s', clkType)
        RETCODE = 1
        return

    point = 0
    if level == 'max':
        point = 1
    try:
        int(clkValue)
    except ValueError:
        printErrLog(None, 'Unable to set %s' % (clkValue))
        logging.exception('%s is not an integer', clkValue)
        RETCODE = 1
        return
    confirmOutOfSpecWarning(autoRespond)
    printLogSpacer(' Set Valid %s Extremum ' % (clkType))
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_clk_extremum_set(
            device,
            rsmi_freq_ind_t(int(point)),
            int(clkValue),
            rsmi_clk_names_dict[clkType],
        )
        if rsmi_ret_ok(ret, device, silent=True):
            printLog(device, f'Successfully set {level} {clkType} to {clkValue}(MHz)', None)
        else:
            printErrLog(device, f'Unable to set {level} {clkType} to {clkValue}(MHz)')
            RETCODE = 1
            if ret == rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED:
                printLog(
                    device,
                    f'Setting {level} {clkType} clock is not supported for this device.',
                    None,
                )


def setVoltageCurve(deviceList, point, clk, volt, autoRespond):
    """Set voltage curve for a point in the PowerPlay table for a list of devices.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param point: Point on the voltage curve to modify
    :param clk: Clock speed specified for this curve point
    :param volt: Voltage specified for this curve point
    :param autoRespond: Response to automatically provide for all prompts
    """
    global RETCODE
    value = f'{point} {clk} {volt}'
    try:
        any(int(item) for item in value.split())
    except ValueError:
        printErrLog(None, 'Unable to set Voltage curve')
        printErrLog(None, 'Non-integer characters are present in %s' % value)
        RETCODE = 1
        return
    confirmOutOfSpecWarning(autoRespond)
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_od_volt_info_set(device, int(point), int(clk), int(volt))
        if rsmi_ret_ok(ret, device, 'set_voltage_curve'):
            printLog(
                device,
                f'Successfully set voltage point {point} to {clk}(MHz) {volt}(mV)',
                None,
            )
        else:
            printErrLog(
                device,
                f'Unable to set voltage point {point} to {clk}(MHz) {volt}(mV)',
            )
            RETCODE = 1


def setPowerPlayTableLevel(deviceList, clkType, point, clk, volt, autoRespond):
    """Set clock frequency and voltage for a level in the PowerPlay table for a list of devices.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param clktype: [sclk|mclk] Which clock type to apply the range to
    :param point: Point on the voltage curve to modify
    :param clk: Clock speed specified for this curve point
    :param volt: Voltage specified for this curve point
    :param autoRespond: Response to automatically provide for all prompts
    """
    global RETCODE
    value = f'{point} {clk} {volt}'
    listOfValues = value.split(' ')
    try:
        any(int(item) for item in value.split())
    except ValueError:
        printErrLog(None, 'Unable to set PowerPlay table level')
        printErrLog(None, 'Non-integer characters are present in %s' % value)
        RETCODE = 1
        return
    confirmOutOfSpecWarning(autoRespond)
    for device in deviceList:
        if clkType == 'sclk' or clkType == 'mclk':
            ret = rocmsmi.rsmi_dev_od_clk_info_set(
                device,
                rsmi_freq_ind_t(int(point)),
                int(clk),
                rsmi_clk_names_dict[clkType],
            )
            if rsmi_ret_ok(ret, device, 'set_power_play_table_level_' + str(clkType)):
                printLog(
                    device,
                    f'Successfully set voltage point {point} to {clk}(MHz) {volt}(mV)',
                    None,
                )
            else:
                printErrLog(
                    device,
                    f'Unable to set voltage point {point} to {clk}(MHz) {volt}(mV)',
                )
                RETCODE = 1
        else:
            printErrLog(device, 'Unable to set %s range' % (clkType))
            logging.error('Unsupported range type %s', clkType)
            RETCODE = 1


def setClockOverDrive(deviceList, clktype, value, autoRespond):
    """Set clock speed to OverDrive for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param type: [sclk|mclk] Clock type to set
    :param value: [0-20] OverDrive percentage
    :param autoRespond: Response to automatically provide for all prompts
    """
    printLogSpacer(' Set Clock OverDrive (Range: 0% to 20%) ')
    global RETCODE
    try:
        int(value)
    except ValueError:
        printLog(None, 'Unable to set OverDrive level', None)
        logging.exception('%s it is not an integer', value)
        RETCODE = 1
        return
    confirmOutOfSpecWarning(autoRespond)
    for device in deviceList:
        if int(value) < 0:
            printErrLog(device, 'Unable to set OverDrive')
            logging.debug('Overdrive cannot be less than 0%')
            RETCODE = 1
            return
        if int(value) > 20:
            printLog(device, 'Setting OverDrive to 20%', None)
            logging.debug('OverDrive cannot be set to a value greater than 20%')
            value = '20'
        if getPerfLevel(device) != 'MANUAL':
            ret = rocmsmi.rsmi_dev_perf_level_set(device, rsmi_dev_perf_level_t(3))
            if rsmi_ret_ok(ret, device, 'set_perf_level_manual_' + str(clktype)):
                printLog(device, 'Performance level set to manual', None)
            else:
                printErrLog(device, 'Unable to set performance level to manual')
        if clktype == 'mclk':
            fsFile = os.path.join('/sys/class/drm', 'card%d' % (device), 'device', 'pp_mclk_od')
            if not os.path.isfile(fsFile):
                printLog(
                    None,
                    'Unable to write to sysfs file (' + fsFile + '), file does not exist',
                    None,
                )
                logging.debug('%s does not exist', fsFile)
                continue
            try:
                logging.debug("Writing value '%s' to file '%s'", value, fsFile)
                with open(fsFile, 'w') as fs:
                    fs.write(value + '\n')
            except OSError:
                printLog(None, 'Unable to write to sysfs file %s' % fsFile, None)
                logging.warning('IO or OS error')
                RETCODE = 1
                continue
            printLog(device, f'Successfully set {clktype} OverDrive to {value}%', None)
        elif clktype == 'sclk':
            ret = rocmsmi.rsmi_dev_overdrive_level_set(device, rsmi_dev_perf_level_t(int(value)))
            if rsmi_ret_ok(ret, device, 'set_overdrive_level_' + str(clktype)):
                printLog(device, f'Successfully set {clktype} OverDrive to {value}%', None)
            else:
                printLog(device, f'Unable to set {clktype} OverDrive to {value}%', None)
        else:
            printErrLog(device, 'Unable to set OverDrive')
            logging.error('Unsupported clock type %s', clktype)
            RETCODE = 1


def setClocks(deviceList, clktype, clk):
    """Set clock frequency levels for a list of devices.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param clktype: [validClockNames] Clock type to set
    :param clk: Clock frequency level to set
    """
    global RETCODE
    if not clk:
        printLog(None, 'Invalid clock frequency', None)
        RETCODE = 1
        return
    if clktype not in validClockNames:
        printErrLog(None, 'Unable to set clock level')
        logging.error('Invalid clock type %s', clktype)
        RETCODE = 1
        return
    check_value = ''.join(map(str, clk))
    try:
        int(check_value)
    except ValueError:
        printLog(None, 'Unable to set clock level', None)
        logging.exception('Non-integer characters are present in value %s', check_value)
        RETCODE = 1
        return
    # Generate a frequency bitmask from user input value
    freq_bitmask = 0
    for bit in clk:
        if bit > 63:
            printErrLog(None, 'Invalid clock frequency')
            logging.error('Invalid frequency: %s', bit)
            RETCODE = 1
            return

        freq_bitmask |= 1 << bit

    printLogSpacer(' Set %s Frequency ' % (str(clktype)))
    for device in deviceList:
        # Check if the performance level is manual, if not then set it to manual
        if getPerfLevel(device).lower() != 'manual':
            ret = rocmsmi.rsmi_dev_perf_level_set(device, rsmi_dev_perf_level_t(3))
            if rsmi_ret_ok(ret, device, 'set_perf_level_manual'):
                printLog(device, 'Performance level was set to manual', None)
            else:
                printErrLog(device, 'Unable to set performance level to manual')
                RETCODE = 1
                return
        if clktype != 'pcie':
            # Validate frequency bitmask
            freq = rsmi_frequencies_t()
            ret = rocmsmi.rsmi_dev_gpu_clk_freq_get(
                device,
                rsmi_clk_names_dict[clktype],
                byref(freq),
            )
            if rsmi_ret_ok(ret, device, 'get_gpu_clk_freq_' + str(clktype)) == False:
                RETCODE = 1
                return
            # The freq_bitmask should be less than 2^(freqs.num_supported)
            # For example, num_supported == 3,  the max bitmask is 0111
            if freq_bitmask >= (1 << freq.num_supported):
                printErrLog(device, 'Invalid clock frequency %s' % hex(freq_bitmask))
                RETCODE = 1
                return

            ret = rocmsmi.rsmi_dev_gpu_clk_freq_set(
                device,
                rsmi_clk_names_dict[clktype],
                freq_bitmask,
            )
            if rsmi_ret_ok(ret, device, 'set_gpu_clk_freq_' + str(clktype)):
                printLog(device, 'Successfully set %s bitmask to' % (clktype), hex(freq_bitmask))
            else:
                printErrLog(
                    device,
                    f'Unable to set {clktype} bitmask to: {hex(freq_bitmask)}',
                )
                RETCODE = 1
        else:
            # Validate the bandwidth bitmask
            bw = rsmi_pcie_bandwidth_t()
            ret = rocmsmi.rsmi_dev_pci_bandwidth_get(device, byref(bw))
            if rsmi_ret_ok(ret, device, 'get_PCIe_bandwidth') == False:
                RETCODE = 1
                return
            # The freq_bitmask should be less than 2^(bw.transfer_rate.num_supported)
            # For example, num_supported == 3,  the max bitmask is 0111
            if freq_bitmask >= (1 << bw.transfer_rate.num_supported):
                printErrLog(device, 'Invalid PCIe frequency %s' % hex(freq_bitmask))
                RETCODE = 1
                return

            ret = rocmsmi.rsmi_dev_pci_bandwidth_set(device, freq_bitmask)
            if rsmi_ret_ok(ret, device, 'set_PCIe_bandwidth'):
                printLog(
                    device,
                    'Successfully set %s to level bitmask' % (clktype),
                    hex(freq_bitmask),
                )
            else:
                printErrLog(
                    device,
                    f'Unable to set {clktype} bitmask to: {hex(freq_bitmask)}',
                )
                RETCODE = 1
    printLogSpacer()


def setPerfDeterminism(deviceList, clkvalue):
    """Set clock frequency level for a list of devices to enable performance
    determinism.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param value: Clock frequency level to set
    """
    global RETCODE
    try:
        int(clkvalue)
    except ValueError:
        printErrLog(None, 'Unable to set Performance Determinism')
        logging.exception('%s is not an integer', clkvalue)
        RETCODE = 1
        return
    for device in deviceList:
        ret = rocmsmi.rsmi_perf_determinism_mode_set(device, int(clkvalue))
        if rsmi_ret_ok(ret, device, 'set_perf_determinism'):
            printLog(
                device,
                'Successfully enabled performance determinism and set GFX clock frequency',
                str(clkvalue),
            )
        else:
            printErrLog(
                device,
                'Unable to set performance determinism and clock frequency to %s' % (str(clkvalue)),
            )
            RETCODE = 1


def resetGpu(device):
    """Perform a GPU reset on the specified device

    :param device: DRM device identifier
    """
    printLogSpacer(' Reset GPU ')
    global RETCODE
    if len(device) > 1:
        logging.error('GPU Reset can only be performed on one GPU per call')
        RETCODE = 1
        return
    resetDev = int(device[0])
    if not isAmdDevice(resetDev):
        logging.error('GPU Reset can only be performed on an AMD GPU')
        RETCODE = 1
        return
    ret = rocmsmi.rsmi_dev_gpu_reset(resetDev)
    if rsmi_ret_ok(ret, resetDev, 'reset_gpu'):
        printLog(resetDev, 'Successfully reset GPU %d' % (resetDev), None)
    else:
        printErrLog(resetDev, 'Unable to reset GPU %d' % (resetDev))
        logging.debug('GPU reset failed with return value of %d' % ret)
    printLogSpacer()


def isRasControlAvailable(device):
    """Check if RAS control is available for a specified device.

    :param device: DRM device identifier
    """

    path = os.path.join('/sys/kernel/debug/dri', 'card%d' % device, 'device', 'ras_ctrl')

    if not doesDeviceExist(device) or not path or not os.path.isfile(path):
        logging.warning('GPU[%s]\t: RAS control is not available')

        return False

    return True


def setRas(deviceList, rasAction, rasBlock, rasType):
    """Perform a RAS action on the devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param rasAction: [enable|disable|inject] RAS Action to perform
    :param rasBlock: [$validRasBlocks] RAS block
    :param rasType: ['ce'|'ue'] Error type to enable/disable
    """
    global RETCODE
    printLog(
        None,
        "This is experimental feature, use 'amdgpuras' tool for ras error manipulations for newer vbios",
    )

    if rasAction not in validRasActions:
        printLog(
            None,
            'Unable to perform RAS command %s on block %s for type %s'
            % (rasAction, rasBlock, rasType),
        )
        logging.debug('Action %s is not a valid RAS command' % rasAction)
        return None
    if rasBlock not in validRasBlocks:
        printLog(
            None,
            'Unable to perform RAS command %s on block %s for type %s'
            % (rasAction, rasBlock, rasType),
        )
        printLog(None, 'Block %s is not a valid RAS block' % rasBlock)
        return None

    if rasType not in validRasTypes:
        printLog(
            None,
            'Unable to perform RAS command %s on block %s for type %s'
            % (rasAction, rasBlock, rasType),
        )
        printLog(None, 'Memory error type %s is not a valid RAS memory type' % rasAction)
        return None

    printLogSpacer()
    # NOTE PSP FW doesn't support enabling disabled counters yet
    for device in deviceList:
        if isRasControlAvailable(device):
            rasFilePath = path = os.path.join(
                '/sys/kernel/debug/dri',
                'card%d' % device,
                'device',
                'ras_ctrl',
            )
            rasCmd = f'{rasAction} {rasBlock} {rasType}'

            # writeToSysfs analog to old cli
            if not os.path.isfile(rasFilePath):
                printLog(None, 'Unable to write to sysfs file', None)
                logging.debug('%s does not exist', rasFilePath)
                return False
            try:
                logging.debug("Writing value '%s' to file '%s'", rasCmd, rasFilePath)
                with open(rasFilePath, 'w') as fs:
                    fs.write(rasFilePath + '\n')  # Certain sysfs files require \n at the end
            except OSError:
                printLog(None, 'Unable to write to sysfs file %s' % rasFilePath, None)
                logging.warning('IO or OS error')
                RETCODE = 1

    printLogSpacer()

    return None


def setFanSpeed(deviceList, fan):
    """Set fan speed for a list of devices.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param level: [0-255] Fan speed level
    """
    printLogSpacer(' Set GPU Fan Speed ')
    for device in deviceList:
        if str(fan):
            fanLevel = c_int64()
            last_char = str(fan)[-1]
            if last_char == '%':
                fanLevel = int(str(fan)[:-1]) / 100 * 255
            else:
                fanLevel = int(str(fan))
            ret = rocmsmi.rsmi_dev_fan_speed_set(device, 0, int(fanLevel))
            if rsmi_ret_ok(ret, device, silent=True):
                printLog(
                    device,
                    'Successfully set fan speed to level %s' % (str(int(fanLevel))),
                    None,
                )
            else:
                printLog(device, 'Not supported on the given system', None)
    printLogSpacer()


def setPerformanceLevel(deviceList, level):
    """Set the Performance Level for a specified device.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param level: Performance Level to set
    """
    printLogSpacer(' Set Performance Level ')
    validLevels = ['auto', 'low', 'high', 'manual']
    for device in deviceList:
        if level not in validLevels:
            printErrLog(device, 'Unable to set Performance Level')
            logging.error('Invalid Performance level: %s', level)
        else:
            ret = rocmsmi.rsmi_dev_perf_level_set(
                device,
                rsmi_dev_perf_level_t(validLevels.index(level)),
            )
            if rsmi_ret_ok(ret, device, 'set_perf_level'):
                printLog(device, 'Performance level set to %s' % (str(level)), None)
    printLogSpacer()


def setPowerOverDrive(deviceList, value, autoRespond):
    """Use Power OverDrive to change the the maximum power available power
    available to the GPU in Watts. May be limited by the maximum power the
    VBIOS is configured to allow this card to use in OverDrive mode.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param value: New maximum power to assign to the target device, in Watts
    :param autoRespond: Response to automatically provide for all prompts
    """
    global RETCODE, PRINT_JSON
    try:
        int(value)
    except ValueError:
        printLog(None, 'Unable to set Power OverDrive', None)
        logging.exception('%s is not an integer', value)
        RETCODE = 1
        return
    # Wattage input value converted to microWatt for ROCm SMI Lib

    if int(value) == 0:
        printLogSpacer(' Reset GPU Power OverDrive ')
    else:
        printLogSpacer(' Set GPU Power OverDrive ')

    # Value in Watts - stored early this way to avoid strenuous value type conversions
    strValue = value
    specWarningConfirmed = False
    for device in deviceList:
        # Continue to next device in deviceList loop if the device is a secondary die
        if checkIfSecondaryDie(device):
            logging.debug('Unavailable for secondary die.')
            continue
        power_cap_min = c_uint64()
        power_cap_max = c_uint64()
        current_power_cap = c_uint64()
        default_power_cap = c_uint64()
        new_power_cap = c_uint64()

        ret = rocmsmi.rsmi_dev_power_cap_get(device, 0, byref(current_power_cap))
        if ret != 0:
            logging.debug('Unable to retireive current power cap.')
        ret = rocmsmi.rsmi_dev_power_cap_default_get(device, byref(default_power_cap))
        # If rsmi_dev_power_cap_default_get fails, use manual workaround to fetch default power cap
        if ret != 0:
            logging.debug('Unable to retrieve default power cap; retrieving via reset.')
            ret = rocmsmi.rsmi_dev_power_cap_set(device, 0, 0)
            ret = rocmsmi.rsmi_dev_power_cap_get(device, 0, byref(default_power_cap))

        if int(value) == 0:
            new_power_cap = default_power_cap
        else:
            new_power_cap.value = int(value) * 1000000

        ret = rocmsmi.rsmi_dev_power_cap_range_get(
            device,
            0,
            byref(power_cap_max),
            byref(power_cap_min),
        )
        if rsmi_ret_ok(ret, device, 'get_power_cap_range') == False:
            printErrLog(device, 'Unable to parse Power OverDrive range')
            RETCODE = 1
            continue
        if int(strValue) > (power_cap_max.value / 1000000):
            printErrLog(device, 'Unable to set Power OverDrive')
            logging.error(
                'GPU[%s]\t\t: Value cannot be greater than: %dW ',
                device,
                power_cap_max.value / 1000000,
            )
            RETCODE = 1
            continue
        if int(strValue) < (power_cap_min.value / 1000000):
            printErrLog(device, 'Unable to set Power OverDrive')
            logging.error(
                'GPU[%s]\t\t: Value cannot be less than: %dW ',
                device,
                power_cap_min.value / 1000000,
            )
            RETCODE = 1
            continue
        if new_power_cap.value == current_power_cap.value:
            printLog(device, f'Max power was already at: {new_power_cap.value / 1000000}W')

        if current_power_cap.value < default_power_cap.value:
            current_power_cap.value = default_power_cap.value
        if not specWarningConfirmed and new_power_cap.value > current_power_cap.value:
            confirmOutOfSpecWarning(autoRespond)
            specWarningConfirmed = True

        ret = rocmsmi.rsmi_dev_power_cap_set(device, 0, new_power_cap)
        if rsmi_ret_ok(ret, device, 'set_power_cap'):
            if int(value) == 0:
                power_cap = c_uint64()
                ret = rocmsmi.rsmi_dev_power_cap_get(device, 0, byref(power_cap))
                if rsmi_ret_ok(ret, device, 'get_power_cap'):
                    if not PRINT_JSON:
                        printLog(
                            device,
                            'Successfully reset Power OverDrive to: %sW'
                            % (int(power_cap.value / 1000000)),
                            None,
                        )
            else:
                if not PRINT_JSON:
                    ret = rocmsmi.rsmi_dev_power_cap_get(device, 0, byref(current_power_cap))
                    if current_power_cap.value == new_power_cap.value:
                        printLog(device, 'Successfully set power to: %sW' % (strValue), None)
                    else:
                        printErrLog(
                            device,
                            'Unable set power to: %sW, current value is %sW'
                            % (strValue, int(current_power_cap.value / 1000000)),
                        )
        else:
            if int(value) == 0:
                printErrLog(device, 'Unable to reset Power OverDrive to default')
            else:
                printErrLog(device, 'Unable to set Power OverDrive to ' + strValue + 'W')
    printLogSpacer()


def setProfile(deviceList, profile):
    """Set Power Profile, or set CUSTOM Power Profile values for a list of devices.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param profile: Profile to set
    """
    printLogSpacer(' Set Power Profile ')
    status = rsmi_power_profile_status_t()
    for device in deviceList:
        # Get previous profile
        ret = rocmsmi.rsmi_dev_power_profile_presets_get(device, 0, byref(status))
        if rsmi_ret_ok(ret, device, 'get_power_profile'):
            previousProfile = profileString(status.current)
            # Get desired profile
            desiredProfile = 'UNKNOWN'
            if str(profile).isnumeric() and int(profile) > 0 and int(profile) < 8:
                desiredProfile = profileString(2 ** (int(profile) - 1))
            elif str(profileString(str(profile).replace('_', ' ').upper())).isnumeric():
                desiredProfile = str(profile).replace('_', ' ').upper()
            else:
                printErrLog(
                    device,
                    'Unable to set profile to: %s (UNKNOWN profile)' % (str(profile)),
                )
                return
            # Set profile to desired profile
            if previousProfile == desiredProfile:
                printLog(device, 'Profile was already set to', previousProfile)
                return
            else:
                ret = rocmsmi.rsmi_dev_power_profile_set(device, 0, profileString(desiredProfile))
                if rsmi_ret_ok(ret, device, 'set_power_profile'):
                    # Get current profile
                    ret = rocmsmi.rsmi_dev_power_profile_presets_get(device, 0, byref(status))
                    if rsmi_ret_ok(ret, device, 'get_power_profile_presets'):
                        currentProfile = profileString(status.current)
                        if currentProfile == desiredProfile:
                            printLog(device, 'Successfully set profile to', desiredProfile)
                        else:
                            printErrLog(device, 'Failed to set profile to: %s' % (desiredProfile))
        printLogSpacer()


def setComputePartition(deviceList, computePartitionType):
    """Sets compute partitioning for a list of device

    :param deviceList: List of DRM devices (can be a single-item list)
    :param computePartition: Compute Partition type to set as
    """
    printLogSpacer(' Set compute partition to %s ' % (str(computePartitionType).upper()))
    for device in deviceList:
        computePartitionType = computePartitionType.upper()
        if computePartitionType not in compute_partition_type_l:
            printErrLog(
                device,
                'Invalid compute partition type %s'
                '\nValid compute partition types are %s'
                % (computePartitionType.upper(), (', '.join(map(str, compute_partition_type_l)))),
            )
            return (None, None)
        ret = rocmsmi.rsmi_dev_compute_partition_set(
            device,
            rsmi_compute_partition_type_dict[computePartitionType],
        )
        if rsmi_ret_ok(ret, device, 'set_compute_partition', silent=True):
            printLog(
                device,
                'Successfully set compute partition to %s' % (computePartitionType),
                None,
            )
        elif ret == rsmi_status_t.RSMI_STATUS_PERMISSION:
            printLog(device, 'Permission denied', None)
        elif ret == rsmi_status_t.RSMI_STATUS_SETTING_UNAVAILABLE:
            printLog(
                device,
                'Requested setting (%s) is unavailable for current device' % computePartitionType,
                None,
            )
        elif ret == rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED:
            printLog(device, 'Not supported on the given system', None)
        elif ret == rsmi_status_t.RSMI_STATUS_BUSY:
            printLog(device, 'Device is currently busy, try again later', None)
        else:
            rsmi_ret_ok(ret, device, 'set_compute_partition')
            printErrLog(
                device,
                'Failed to retrieve compute partition, even though device supports it.',
            )
    printLogSpacer()


def progressbar(it, prefix='', size=60, out=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        lock = multiprocessing.Lock()
        lock.acquire()
        print(
            '{}[{}{}] {}/{} secs remain'.format(prefix, '█' * x, '.' * (size - x), j, count),
            end='\r',
            file=out,
            flush=True,
        )
        lock.release()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    lock = multiprocessing.Lock()
    lock.acquire()
    print('\n', flush=True, file=out)
    lock.release()


def showProgressbar(title='', timeInSeconds=13):
    if title != '':
        title += ': '
    for i in progressbar(range(timeInSeconds), title, 40):
        time.sleep(1)


def setMemoryPartition(deviceList, memoryPartition):
    """Sets memory partition (memory partition) for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param memoryPartition: Memory Partition type to set as
    """
    printLogSpacer(' Set memory partition to %s ' % (str(memoryPartition).upper()))
    for device in deviceList:
        memoryPartition = memoryPartition.upper()
        if memoryPartition not in memory_partition_type_l:
            printErrLog(
                device,
                'Invalid memory partition type %s'
                '\nValid memory partition types are %s'
                % (memoryPartition.upper(), (', '.join(map(str, memory_partition_type_l)))),
            )
            return (None, None)

        t1 = multiprocessing.Process(target=showProgressbar, args=('Updating memory partition', 13))
        t1.start()
        addExtraLine = True
        start = time.time()
        ret = rocmsmi.rsmi_dev_memory_partition_set(
            device,
            rsmi_memory_partition_type_dict[memoryPartition],
        )
        stop = time.time()
        duration = stop - start
        if t1.is_alive():
            t1.terminate()
            t1.join()
        if duration < 0.1:  # For longer runs, add extra line before output
            addExtraLine = False  # This is to prevent overriding progress bar

        if rsmi_ret_ok(ret, device, 'set_memory_partition', silent=True):
            printLog(
                device,
                'Successfully set memory partition to %s' % (memoryPartition),
                None,
                addExtraLine,
            )
        elif ret == rsmi_status_t.RSMI_STATUS_PERMISSION:
            printLog(device, 'Permission denied', None, addExtraLine)
        elif ret == rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED:
            printLog(device, 'Not supported on the given system', None, addExtraLine)
        elif ret == rsmi_status_t.RSMI_STATUS_BUSY:
            printLog(device, 'Device is currently busy, try again later', None, addExtraLine)
        else:
            rsmi_ret_ok(ret, device, 'set_memory_partition')
            printErrLog(
                device,
                'Failed to retrieve memory partition, even though device supports it.',
            )
    printLogSpacer()


def showVersion(isCSV=False):
    values = {'ROCM-SMI version': __version__}

    version = rsmi_version_t()
    status = rocmsmi.rsmi_version_get(byref(version))
    if status == 0:
        version_string = '%u.%u.%u' % (version.major, version.minor, version.patch)
        values['ROCM-SMI-LIB version'] = version_string

    if isCSV:
        print('name, value')
        for k in values:
            print(f'{k}, {values[k]}')
        return
    if PRINT_JSON:
        temp_str = '{\n'
        for k in values:
            temp_str += f'  "{k}": "{values[k]}",\n'
        if len(values.keys()) > 1:
            # replace ',\n' with '\n}'
            temp_str = temp_str[:-2]
        temp_str += '\n}'
        print(temp_str)
        return
    for k in values:
        print(f'{k}: {values[k]}')


def showAllConcise(deviceList):
    """Display critical info for all devices in a concise format

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    global PRINT_JSON, appWidth
    if PRINT_JSON:
        print('ERROR: Cannot print JSON/CSV output for concise output')
        sys.exit(1)

    silent = True

    deviceList.sort()
    available_temp_type = getTemperatureLabel(deviceList)
    temp_type = '(' + available_temp_type.capitalize() + ')'
    header = [
        'Device',
        'Node',
        'IDs',
        '',
        'Temp',
        'Power',
        'Partitions',
        'SCLK',
        'MCLK',
        'Fan',
        'Perf',
        'PwrCap',
        'VRAM%',
        'GPU%',
    ]
    subheader = [
        '',
        '',
        '(DID,',
        'GUID)',
        temp_type,
        getPowerLabel(deviceList),
        '(Mem, Compute, ID)',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
    ]
    # add additional spaces to match header
    for idx, item in enumerate(subheader):
        header_size = len(header[idx])
        subheader_size = len(subheader[idx])
        if header_size != subheader_size:
            numSpacesToFill_subheader = header_size - subheader_size
            numSpacesToFill_header = subheader_size - header_size
            # take pos spaces to mean, we need to match size of the other
            if numSpacesToFill_subheader > 0:
                subheader[idx] = subheader[idx] + (' ' * numSpacesToFill_subheader)
            if numSpacesToFill_header > 0:
                header[idx] = header[idx] + (' ' * numSpacesToFill_header)
    head_widths = [len(head) + 2 for head in header]
    values = {}
    degree_sign = '\N{DEGREE SIGN}'
    for device in deviceList:
        temp_val = str(getTemp(device, available_temp_type, silent))
        if temp_val != 'N/A':
            temp_val += degree_sign + 'C'
        power_dict = getPower(device)
        powerVal = 'N/A'
        if (
            power_dict['ret'] == rsmi_status_t.RSMI_STATUS_SUCCESS
            and power_dict['power_type'] != 'INVALID_POWER_TYPE'
        ):
            if power_dict['power'] != 0:
                powerVal = power_dict['power'] + power_dict['unit']
        combined_partition_data = (
            getMemoryPartition(device, silent)
            + ', '
            + getComputePartition(device, silent)
            + ', '
            + getPartitionId(device, silent)
        )
        sclk = showCurrentClocks([device], 'sclk', concise=silent)
        mclk = showCurrentClocks([device], 'mclk', concise=silent)
        (retCode, fanLevel, fanSpeed) = getFanSpeed(device, silent)
        fan = str(fanSpeed) + '%'
        if getPerfLevel(device, silent) != -1:
            perf = getPerfLevel(device, silent)
        else:
            perf = 'Unsupported'
        if getMaxPower(device, silent) != -1:
            pwrCap = str(getMaxPower(device, silent)) + 'W'
        else:
            pwrCap = 'Unsupported'
        if getGpuUse(device, silent) != -1:
            gpu_busy = str(getGpuUse(device, silent)) + '%'
        else:
            gpu_busy = 'Unsupported'
        vram_used, vram_total = getMemInfo(device, 'vram', silent)
        mem_use_pct = 0
        if vram_used is None:
            mem_use_pct = 'Unsupported'
        if vram_used != None and vram_total != None and float(vram_total) != 0:
            mem_use_pct = round(float(100 * (float(vram_used) / float(vram_total))))
            mem_use_pct = f'{mem_use_pct:<.0f}%'  # left aligned
            # values with no precision

        # Top Row - per device data
        values['card%s' % (str(device))] = [
            device,
            getNodeId(device),
            str(getDRMDeviceId(device)) + ', ',
            str(getGUID(device)),
            temp_val,
            powerVal,
            combined_partition_data,
            sclk,
            mclk,
            fan,
            str(perf).lower(),
            str(pwrCap),
            str(mem_use_pct),
            str(gpu_busy),
        ]

    val_widths = {}
    for device in deviceList:
        val_widths[device] = [len(str(val)) + 2 for val in values['card%s' % (str(device))]]
    max_widths = head_widths
    for device in deviceList:
        for col in range(len(val_widths[device])):
            max_widths[col] = max(max_widths[col], val_widths[device][col])

    ########################
    # Display concise info #
    ########################
    header_output = ''.join(
        word.ljust(max_widths[col]) for col, word in zip(range(len(max_widths)), header)
    )
    subheader_output = ''.join(
        word.ljust(max_widths[col]) for col, word in zip(range(len(max_widths)), subheader)
    )
    printLogSpacer(headerString, contentSizeToFit=len(header_output))
    printLogSpacer(' Concise Info ', contentSizeToFit=len(header_output))
    printLog(None, header_output, None)
    printLog(None, subheader_output, None, useItalics=True)
    printLogSpacer(fill='=', contentSizeToFit=len(header_output))

    for device in deviceList:
        printLog(
            None,
            ''.join(
                str(word).ljust(max_widths[col])
                for col, word in zip(range(len(max_widths)), values['card%s' % (str(device))])
            ),
            None,
        )

    printLogSpacer(contentSizeToFit=len(header_output))
    printLogSpacer(footerString, contentSizeToFit=len(header_output))


def showAllConciseHw(deviceList):
    """Display critical Hardware info

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    global PRINT_JSON
    if PRINT_JSON:
        print('ERROR: Cannot print JSON/CSV output for concise hardware output')
        sys.exit(1)
    header = [
        'GPU',
        'NODE',
        'DID',
        'GUID',
        'GFX VER',
        'GFX RAS',
        'SDMA RAS',
        'UMC RAS',
        'VBIOS',
        'BUS',
        'PARTITION ID',
    ]
    head_widths = [len(head) + 2 for head in header]
    values = {}
    silent = True
    for device in deviceList:
        did = getDRMDeviceId(device, silent)
        nodeid = getNodeId(device, silent)
        guid = getGUID(device, silent)
        partition_id = getPartitionId(device, silent)
        gfxVer = getTargetGfxVersion(device, silent)
        gfxRas = getRasEnablement(device, 'GFX', silent)
        sdmaRas = getRasEnablement(device, 'SDMA', silent)
        umcRas = getRasEnablement(device, 'UMC', silent)
        vbios = getVbiosVersion(device, silent)
        bus = getBus(device, silent)
        values['card%s' % (str(device))] = [
            device,
            nodeid,
            did,
            guid,
            gfxVer,
            gfxRas,
            sdmaRas,
            umcRas,
            vbios,
            bus,
            partition_id,
        ]
    val_widths = {}
    for device in deviceList:
        val_widths[device] = [len(str(val)) + 2 for val in values['card%s' % (str(device))]]
    max_widths = head_widths
    for device in deviceList:
        for col in range(len(val_widths[device])):
            max_widths[col] = max(max_widths[col], val_widths[device][col])
    device_output = ''
    for device in deviceList:
        if device + 1 != len(deviceList):
            device_output += (
                ''.join(
                    str(word).ljust(max_widths[col])
                    for col, word in zip(range(len(max_widths)), values['card%s' % (str(device))])
                )
                + '\n'
            )
        else:
            device_output += ''.join(
                str(word).ljust(max_widths[col])
                for col, word in zip(range(len(max_widths)), values['card%s' % (str(device))])
            )

    #################################
    # Display concise hardware info #
    #################################
    header_output = ''.join(
        word.ljust(max_widths[col]) for col, word in zip(range(len(max_widths)), header)
    )
    printLogSpacer(headerString, contentSizeToFit=len(header_output))
    printLogSpacer(' Concise Hardware Info ', contentSizeToFit=len(header_output))
    printLog(None, header_output, None)
    printLog(None, device_output, None)
    printLogSpacer(fill='=', contentSizeToFit=len(header_output))
    printLogSpacer(footerString, contentSizeToFit=len(header_output))


def showBus(deviceList):
    """Display PCI Bus info

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' PCI Bus ID ')
    for device in deviceList:
        printLog(device, 'PCI Bus', getBus(device))
    printLogSpacer()


def showClocks(deviceList):
    """Display all available clocks for a list of devices

    Current clocks marked with a '*' symbol

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    freq = rsmi_frequencies_t()
    bw = rsmi_pcie_bandwidth_t()
    printLogSpacer(' Supported clock frequencies ')
    for device in deviceList:
        for clk_type in sorted(rsmi_clk_names_dict):
            if rocmsmi.rsmi_dev_gpu_clk_freq_get(device, rsmi_clk_names_dict[clk_type], None) == 1:
                ret = rocmsmi.rsmi_dev_gpu_clk_freq_get(
                    device,
                    rsmi_clk_names_dict[clk_type],
                    byref(freq),
                )
                if ret == rsmi_status_t.RSMI_STATUS_UNEXPECTED_DATA:
                    printLog(
                        device,
                        'Clock [%s] on device [%s] exists but EMPTY! Likely driver error!'
                        % (clk_type, str(device)),
                    )
                    continue
                if not rsmi_ret_ok(ret, device, 'get_clk_freq_' + clk_type, True):
                    continue
                printLog(
                    device,
                    f'Supported {clk_type} frequencies on GPU{device!s}',
                    None,
                )
                for i in range(freq.num_supported):
                    freq_string = f'{freq.frequency[i] / 1000000:>.0f}Mhz'
                    if i == freq.current:
                        freq_string += ' *'
                    freq_index = i
                    # Deep Sleep frequency is only supported by some GPUs
                    # It is indicated by letter 'S' instead of the index number
                    if freq.has_deep_sleep:
                        # sleep state
                        if i == 0:
                            freq_index = 'S'
                        # all indices are offset by 1 because Deep Sleep occupies index 0
                        else:
                            freq_index = i - 1
                    printLog(device, str(freq_index), freq_string)
                printLog(device, '', None)
            else:
                logging.debug(f'{clk_type} frequency is unsupported on device[{device}]')
                printLog(device, '', None)
        if rocmsmi.rsmi_dev_pci_bandwidth_get(device, None) == 1:
            ret = rocmsmi.rsmi_dev_pci_bandwidth_get(device, byref(bw))
            if rsmi_ret_ok(ret, device, 'get_PCIe_bandwidth', True):
                printLog(
                    device,
                    'Supported {} frequencies on GPU{}'.format('PCIe', str(device)),
                    None,
                )
                for i in range(bw.transfer_rate.num_supported):
                    freq_string = (
                        f'{bw.transfer_rate.frequency[i] / 1000000000:>.1f}GT/s x{bw.lanes[i]}'
                    )
                    if i == bw.transfer_rate.current:
                        freq_string += ' *'
                    printLog(device, str(i), str(freq_string))
                printLog(device, '', None)
        else:
            logging.debug(f'PCIe frequency is unsupported on device [{device}]')
            printLog(device, '', None)
        printLogSpacer(None, '-')  # divider between devices for better visibility
    printLogSpacer()


def showCurrentClocks(deviceList, clk_defined=None, concise=False):
    """Display all clocks for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param clk-type: Clock type to display
    """
    global PRINT_JSON
    freq = rsmi_frequencies_t()
    bw = rsmi_pcie_bandwidth_t()
    if not concise:
        printLogSpacer(' Current clock frequencies ')
    for device in deviceList:
        if clk_defined:
            if (
                rocmsmi.rsmi_dev_gpu_clk_freq_get(device, rsmi_clk_names_dict[clk_defined], None)
                == 1
            ):
                ret = rocmsmi.rsmi_dev_gpu_clk_freq_get(
                    device,
                    rsmi_clk_names_dict[clk_defined],
                    byref(freq),
                )
                if rsmi_ret_ok(ret, device, 'get_gpu_clk_freq_' + str(clk_defined), silent=True):
                    level = freq.current
                    if level >= freq.num_supported:
                        printLog(
                            device,
                            '%s current clock frequency not found' % (clk_defined),
                            None,
                        )
                        continue
                    fr = freq.frequency[level] / 1000000
                    freq_index = level
                    if freq.has_deep_sleep:
                        # sleep state
                        if level == 0:
                            freq_index = 'S'
                        # all indices are offset by 1 because Deep Sleep occupies index 0
                        else:
                            freq_index = level - 1
                    if concise:  # in case function is used for concise output, no need to print.
                        return f'{fr:.0f}Mhz'
                    printLog(device, f'{clk_defined} clock level', f'{freq_index} ({fr:.0f}Mhz)')
            elif not concise:
                logging.debug(f'{clk_defined} clock is unsupported on device[{device}]')

        else:  # if clk is not defined, will display all current clk
            for clk_type in sorted(rsmi_clk_names_dict):
                if (
                    rocmsmi.rsmi_dev_gpu_clk_freq_get(device, rsmi_clk_names_dict[clk_type], None)
                    == 1
                ):
                    ret = rocmsmi.rsmi_dev_gpu_clk_freq_get(
                        device,
                        rsmi_clk_names_dict[clk_type],
                        byref(freq),
                    )
                    if rsmi_ret_ok(ret, device, 'get_clk_freq_' + str(clk_type), True):
                        level = freq.current
                        if level >= freq.num_supported:
                            printLog(
                                device,
                                '%s current clock frequency not found' % (clk_type),
                                None,
                            )
                            continue
                        freq_index = level
                        if freq.has_deep_sleep:
                            # sleep state
                            if level == 0:
                                freq_index = 'S'
                            # all indices are offset by 1 because Deep Sleep occupies index 0
                            else:
                                freq_index = level - 1
                        fr = freq.frequency[level] / 1000000
                        if PRINT_JSON:
                            printLog(
                                device,
                                '%s clock speed:' % (clk_type),
                                '(%sMhz)' % (str(fr)[:-2]),
                            )
                            printLog(device, '%s clock level:' % (clk_type), freq_index)
                        else:
                            printLog(
                                device,
                                f'{clk_type} clock level: {freq_index}',
                                '(%sMhz)' % (str(fr)[:-2]),
                            )
                elif not concise:
                    logging.debug(f'{clk_type} clock is unsupported on device[{device}]')
            # pcie clocks
            if rocmsmi.rsmi_dev_pci_bandwidth_get(device, None) == 1:
                ret = rocmsmi.rsmi_dev_pci_bandwidth_get(device, byref(bw))
                if rsmi_ret_ok(ret, device, 'get_PCIe_bandwidth', True):
                    current_f = bw.transfer_rate.current
                    if current_f >= bw.transfer_rate.num_supported:
                        printLog(device, 'PCIe current clock frequency not found', None)
                        continue
                    fr = '{:.1f}GT/s x{}'.format(
                        bw.transfer_rate.frequency[current_f] / 1000000000,
                        bw.lanes[current_f],
                    )
                    printLog(device, 'pcie clock level', f'{current_f} ({fr})')
            elif not concise:
                logging.debug('{} clock is unsupported on device[{}]'.format('PCIe', device))
    if not concise:
        printLogSpacer()


def showCurrentFans(deviceList):
    """Display the current fan speed for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    global PRINT_JSON
    printLogSpacer(' Current Fan Metric ')
    rpmSpeed = c_int64()
    sensor_ind = c_uint32(0)

    for device in deviceList:
        (retCode, fanLevel, fanSpeed) = getFanSpeed(device)
        if retCode == rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED:
            printLog(device, 'Not supported', None)
        else:
            fanSpeed = round(fanSpeed)
            if fanLevel == 0 or fanSpeed == 0:
                printLog(device, 'Unable to detect fan speed for GPU %d' % (device), None)
                logging.debug(
                    'Current fan speed is: %d\n' % (fanSpeed)
                    + '       Current fan level is: %d\n' % (fanLevel)
                    + '       (GPU might be cooled with a non-PWM fan)',
                )
                continue
            if PRINT_JSON:
                printLog(device, 'Fan speed (level)', str(fanLevel))
                printLog(device, 'Fan speed (%)', str(fanSpeed))
            else:
                printLog(device, 'Fan Level', str(fanLevel) + ' (%s%%)' % (str(fanSpeed)))
            ret = rocmsmi.rsmi_dev_fan_rpms_get(device, sensor_ind, byref(rpmSpeed))
            if rsmi_ret_ok(ret, device, 'get_fan_rpms'):
                printLog(device, 'Fan RPM', rpmSpeed.value)
    printLogSpacer()


def showCurrentTemps(deviceList):
    """Display all available temperatures for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Temperature ')
    for device in deviceList:
        for sensor in temp_type_lst:
            temp = getTemp(device, sensor)
            if temp != 'N/A':
                printLog(device, 'Temperature (Sensor %s) (C)' % (sensor), temp)
            else:
                printInfoLog(device, 'Temperature (Sensor %s) (C)' % (sensor), temp)
    printLogSpacer()


def showFwInfo(deviceList, fwType):
    """Show the requested FW information for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param fwType: [$validFwBlocks] FW block version to display (all if left empty)
    """
    if not fwType or 'all' in fwType:
        firmware_blocks = fw_block_names_l
    else:
        for name in fwType:  # cleaning list from wrong values
            if name.upper() not in fw_block_names_l:
                fwType.remove(name)
        firmware_blocks = fwType
    printLogSpacer(' Firmware Information ')
    for device in deviceList:
        fw_ver = c_uint64()
        for fw_name in firmware_blocks:
            fw_name = fw_name.upper()
            ret = rocmsmi.rsmi_dev_firmware_version_get(
                device,
                fw_block_names_l.index(fw_name),
                byref(fw_ver),
            )
            if rsmi_ret_ok(ret, device, 'get_firmware_version_' + str(fw_name)):
                # The VCN, VCE, UVD, SOS and ASD firmware's value needs to be in hexadecimal
                if fw_name in ['VCN', 'VCE', 'UVD', 'SOS', 'ASD', 'MES', 'MES KIQ']:
                    printLog(
                        device,
                        '%s firmware version' % (fw_name),
                        '\t0x%s' % (str(hex(fw_ver.value))[2:].zfill(8)),
                    )
                # The TA XGMI, TA RAS, and SMC firmware's hex value looks like 0x12345678
                # However, they are parsed as: int(0x12).int(0x34).int(0x56).int(0x78)
                # Which results in the following: 12.34.56.78
                elif fw_name in ['TA XGMI', 'TA RAS', 'SMC']:
                    pos1 = str(
                        '%02d' % int(('0x%s' % (str(hex(fw_ver.value))[2:].zfill(8))[0:2]), 16),
                    )
                    pos2 = str(
                        '%02d' % int(('0x%s' % (str(hex(fw_ver.value))[2:].zfill(8))[2:4]), 16),
                    )
                    pos3 = str(
                        '%02d' % int(('0x%s' % (str(hex(fw_ver.value))[2:].zfill(8))[4:6]), 16),
                    )
                    pos4 = str(
                        '%02d' % int(('0x%s' % (str(hex(fw_ver.value))[2:].zfill(8))[6:8]), 16),
                    )
                    printLog(
                        device,
                        '%s firmware version' % (fw_name),
                        f'\t{pos1}.{pos2}.{pos3}.{pos4}',
                    )
                # The ME, MC, and CE firmware names are only 2 characters, so they need an additional tab
                elif fw_name in ['ME', 'MC', 'CE']:
                    printLog(
                        device,
                        '%s firmware version' % (fw_name),
                        '\t\t%s' % (str(fw_ver.value)),
                    )
                else:
                    printLog(
                        device,
                        '%s firmware version' % (fw_name),
                        '\t%s' % (str(fw_ver.value)),
                    )
    printLogSpacer()


def showGpusByPid(pidList):
    """Show GPUs used by a specific Process ID (pid)

    Print out the GPU(s) used by a specific KFD process
    If pidList is empty, print all used GPUs for all KFD processes

    :param pidList: List of PIDs to check
    """
    printLogSpacer(' GPUs Indexed by PID ')
    # If pidList is empty then we were given 0 arguments, so they want all PIDs
    # dv_indices = (c_uint32 * dv_limit)()
    num_devices = c_uint32()
    dv_indices = c_void_p()

    if not pidList:
        pidList = getPidList()
        if not pidList:
            printLog(None, 'No KFD PIDs currently running', None)
            printLogSpacer()
            return
    for pid in pidList:
        ret = rocmsmi.rsmi_compute_process_gpus_get(int(pid), None, byref(num_devices))
        if rsmi_ret_ok(ret, metric=('PID ' + pid)):

            dv_indices = (c_uint32 * num_devices.value)()
            ret = rocmsmi.rsmi_compute_process_gpus_get(int(pid), dv_indices, byref(num_devices))

            if rsmi_ret_ok(ret, metric='get_gpu_compute_process'):
                metricName = f'PID {pid} is using {num_devices.value!s} DRM device(s)'
                if num_devices.value:
                    printListLog(metricName, list(dv_indices))
                else:
                    printLog(None, metricName, None)
        else:
            print(None, 'Unable to get list of KFD PIDs. A kernel update may be needed', None)
    printLogSpacer()


def getCoarseGrainUtil(device, typeName=None):
    """Find Coarse Grain Utilization
        If typeName is not given, will return array with of all available sensors,
        where sensor type and value could be addressed like this:

         .. code-block:: python

            for ut_counter in utilization_counters:
                printLog(device, utilization_counter_name[ut_counter.type], ut_counter.val)

    :param device: DRM device identifier
    :param typeName: 'GFX Activity', 'Memory Activity'
    """
    timestamp = c_uint64(0)

    if typeName != None:

        try:
            i = utilization_counter_name.index(typeName)
            length = 1
            utilization_counters = (rsmi_utilization_counter_t * length)()
            utilization_counters[0].type = c_int(i)
        except ValueError:
            printLog(None, 'No such coarse grain counter type')
            return -1

    else:
        length = rsmi_utilization_counter_type.RSMI_UTILIZATION_COUNTER_LAST + 1
        utilization_counters = (rsmi_utilization_counter_t * length)()
        # populate array with all existing types to query
        for i in range(length):
            utilization_counters[i].type = c_int(i)

    ret = rocmsmi.rsmi_utilization_count_get(device, utilization_counters, length, byref(timestamp))
    if rsmi_ret_ok(ret, device, 'get_utilization_count_' + str(typeName), True):
        return utilization_counters
    return -1


def showGpuUse(deviceList):
    """Display GPU use for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' % time GPU is busy ')
    for device in deviceList:
        if getGpuUse(device) != -1:
            printLog(device, 'GPU use (%)', getGpuUse(device))
        else:
            printLog(device, 'GPU use Unsupported', None)
        util_counters = getCoarseGrainUtil(device, 'GFX Activity')
        if util_counters != -1:
            for ut_counter in util_counters:
                printLog(device, utilization_counter_name[ut_counter.type], ut_counter.val)
        else:
            printInfoLog(device, 'GFX Activity', 'N/A')

    printLogSpacer()


def showEnergy(deviceList):
    """Display amount of energy consumed by device until now

    Default counter value is 10000b, indicating energy status unit
    is 15.3 micro-Joules increment.
    :param deviceList: List of DRM devices (can be a single-item list)
    """
    power = c_uint64()
    timestamp = c_uint64()
    counter_resolution = c_float()
    printLogSpacer(' Consumed Energy ')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_energy_count_get(
            device,
            byref(power),
            byref(counter_resolution),
            byref(timestamp),
        )
        if rsmi_ret_ok(ret, device, '% Energy Counter'):
            printLog(device, 'Energy counter', power.value)
            printLog(
                device,
                'Accumulated Energy (uJ)',
                round(power.value * counter_resolution.value, 2),
            )
    printLogSpacer()


def showId(deviceList):
    """Display the device IDs for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' ID ')
    for device in deviceList:
        printLog(device, 'Device Name', '\t\t' + str(getDeviceName(device)))
        printLog(device, 'Device ID', '\t\t' + str(getDRMDeviceId(device)))
        printLog(device, 'Device Rev', '\t\t' + str(getRev(device)))
        printLog(device, 'Subsystem ID', '\t' + str(getSubsystemId(device)))
        printLog(device, 'GUID', '\t\t' + str(getGUID(device)))
    printLogSpacer()


def showMaxPower(deviceList):
    """Display the maximum Graphics Package Power that this GPU will attempt to consume
    before it begins throttling performance

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Power Cap ')
    for device in deviceList:
        if getMaxPower(device) != -1:
            printLog(device, 'Max Graphics Package Power (W)', getMaxPower(device))
        else:
            printLog(device, 'Max Graphics Package Power Unsupported', None)
    printLogSpacer()


def showMemInfo(deviceList, memType):
    """Display Memory information for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param memType: [$validMemTypes] Type of memory information to display
    """
    # Python will pass in a list of values as a single-value list
    # If we get 'all' as the string, just set the list to all supported types
    # Otherwise, split the single-item list by space, then split each element
    # up to process it below

    if 'all' in memType:
        returnTypes = memory_type_l
    else:
        returnTypes = memType

    printLogSpacer(' Memory Usage (Bytes) ')
    for device in deviceList:
        for mem in returnTypes:
            mem = mem.upper()
            memInfo = getMemInfo(device, mem)
            printLog(device, '%s Total Memory (B)' % (mem), memInfo[1])
            printLog(device, '%s Total Used Memory (B)' % (mem), memInfo[0])
    printLogSpacer()


def showMemUse(deviceList):
    """Display GPU memory usage for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    memoryUse = c_uint64()
    avgMemBandwidth = c_uint16()
    printLogSpacer(' Current Memory Use ')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_memory_busy_percent_get(device, byref(memoryUse))
        if rsmi_ret_ok(ret, device, '% memory use'):
            printLog(device, 'GPU memory use (%)', memoryUse.value)
        util_counters = getCoarseGrainUtil(device, 'Memory Activity')
        if util_counters != -1:
            for ut_counter in util_counters:
                printLog(device, utilization_counter_name[ut_counter.type], ut_counter.val)
        else:
            printLog(device, 'Memory Activity', 'N/A')

        ret = rocmsmi.rsmi_dev_activity_avg_mm_get(device, byref(avgMemBandwidth))
        if rsmi_ret_ok(ret, device, silent=True):
            printLog(device, 'Avg. Memory Bandwidth', avgMemBandwidth.value)
        else:
            printLog(device, 'Not supported on the given system', None)
    printLogSpacer()


def showMemVendor(deviceList):
    """Display GPU memory vendor for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    vendor = create_string_buffer(256)
    printLogSpacer(' Memory Vendor ')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_vram_vendor_get(device, vendor, 256)
        try:
            if rsmi_ret_ok(ret, device, 'get_vram_vendor') and vendor.value.decode():
                printLog(device, 'GPU memory vendor', vendor.value.decode())
            else:
                logging.debug('GPU memory vendor missing or not supported')
        except UnicodeDecodeError:
            printErrLog(device, 'Unable to read GPU memory vendor')
    printLogSpacer()


def showOverDrive(deviceList, odtype):
    """Display current OverDrive level for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param odtype: [sclk|mclk] OverDrive type
    """
    rsmi_od = c_uint32()
    printLogSpacer(' OverDrive Level ')
    for device in deviceList:
        odStr = ''
        od = ''
        if odtype == 'sclk':
            odStr = 'GPU'
            ret = rocmsmi.rsmi_dev_overdrive_level_get(device, byref(rsmi_od))
            od = rsmi_od.value
            if not rsmi_ret_ok(ret, device, 'get_overdrive_level_' + str(odtype)):
                continue
        elif odtype == 'mclk':
            odStr = 'GPU Memory'
            ret = rocmsmi.rsmi_dev_mem_overdrive_level_get(device, byref(rsmi_od))
            od = rsmi_od.value
            if not rsmi_ret_ok(ret, device, 'get_mem_overdrive_level_' + str(odtype)):
                continue
        else:
            printErrLog(device, 'Unable to retrieve OverDrive')
            logging.error('Unsupported clock type %s', odtype)
        printLog(device, odStr + ' OverDrive value (%)', od)
    printLogSpacer()


def showPcieBw(deviceList):
    """Display estimated PCIe bandwidth usage for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    sent = c_uint64()
    received = c_uint64()
    max_pkt_sz = c_uint64()
    printLogSpacer(' Measured PCIe Bandwidth ')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_pci_throughput_get(
            device,
            byref(sent),
            byref(received),
            byref(max_pkt_sz),
        )
        if rsmi_ret_ok(ret, device, 'get_PCIe_bandwidth'):
            # Use 1024.0 to ensure that the result is a float and not integer division
            bw = ((received.value + sent.value) * max_pkt_sz.value) / 1024.0 / 1024.0
            # Use the bwstr below to control precision on the string
            bwstr = '%.3f' % bw
            printLog(device, 'Estimated maximum PCIe bandwidth over the last second (MB/s)', bwstr)
        else:
            logging.debug('GPU PCIe bandwidth usage not supported')
    printLogSpacer()


def showPcieReplayCount(deviceList):
    """Display number of PCIe replays for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    counter = c_uint64()
    printLogSpacer(' PCIe Replay Counter ')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_pci_replay_counter_get(device, byref(counter))
        if rsmi_ret_ok(ret, device, 'PCIe Replay Count'):
            printLog(device, 'PCIe Replay Count', counter.value)
    printLogSpacer()


def showPerformanceLevel(deviceList):
    """Display current Performance Level for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Show Performance Level ')
    for device in deviceList:
        if getPerfLevel(device) != -1:
            printLog(device, 'Performance Level', str(getPerfLevel(device)).lower())
        else:
            printLog(device, 'Performance Level Unsupported', None)
    printLogSpacer()


def showPids(verbose):
    """Show Information for PIDs created in a KFD (Compute) context"""
    printLogSpacer(' KFD Processes ')
    dataArray = []
    if verbose == 'details':
        dataArray.append(['PID', 'PROCESS NAME', 'GPU', 'VRAM USED', 'SDMA USED', 'CU OCCUPANCY'])
    else:
        dataArray.append(
            ['PID', 'PROCESS NAME', 'GPU(s)', 'VRAM USED', 'SDMA USED', 'CU OCCUPANCY'],
        )

    pidList = getPidList()
    if not pidList:
        printLog(None, 'No KFD PIDs currently running', None)
        printLogSpacer()
        return
    dv_indices = c_void_p()
    num_devices = c_uint32()
    proc = rsmi_process_info_t()
    for pid in pidList:
        gpuNumber = 'UNKNOWN'
        vramUsage = 'UNKNOWN'
        sdmaUsage = 'UNKNOWN'
        cuOccupancy = 'UNKNOWN'
        cuOccupancyInvalid = 0xFFFFFFFF
        dv_indices = (c_uint32 * num_devices.value)()
        ret = rocmsmi.rsmi_compute_process_gpus_get(int(pid), None, byref(num_devices))
        if rsmi_ret_ok(ret, metric='get_gpu_compute_process'):
            dv_indices = (c_uint32 * num_devices.value)()
            ret = rocmsmi.rsmi_compute_process_gpus_get(int(pid), dv_indices, byref(num_devices))
            if rsmi_ret_ok(ret, metric='get_gpu_compute_process'):
                gpuNumber = str(num_devices.value)
            else:
                logging.debug('Unable to fetch GPU number by PID')
        if verbose == 'details':
            for dv_ind in dv_indices:
                ret = rocmsmi.rsmi_compute_process_info_by_device_get(int(pid), dv_ind, byref(proc))
                if rsmi_ret_ok(ret, metric='get_compute_process_info_by_pid'):
                    vramUsage = proc.vram_usage
                    sdmaUsage = proc.sdma_usage
                    if proc.cu_occupancy != cuOccupancyInvalid:
                        cuOccupancy = proc.cu_occupancy
                else:
                    logging.debug('Unable to fetch process info by PID')
                dataArray.append(
                    [
                        pid,
                        getProcessName(pid),
                        str(gpuNumber),
                        str(vramUsage),
                        str(sdmaUsage),
                        str(cuOccupancy),
                    ],
                )
        else:
            ret = rocmsmi.rsmi_compute_process_info_by_pid_get(int(pid), byref(proc))
            if rsmi_ret_ok(ret, metric='get_compute_process_info_by_pid'):
                vramUsage = proc.vram_usage
                sdmaUsage = proc.sdma_usage
                if proc.cu_occupancy != cuOccupancyInvalid:
                    cuOccupancy = proc.cu_occupancy
            else:
                logging.debug('Unable to fetch process info by PID')
            dataArray.append(
                [
                    pid,
                    getProcessName(pid),
                    str(gpuNumber),
                    str(vramUsage),
                    str(sdmaUsage),
                    str(cuOccupancy),
                ],
            )
    printLog(None, 'KFD process information:', None)
    print2DArray(dataArray)
    printLogSpacer()


def showPower(deviceList):
    """Display Current (also known as instant) Socket or Average
        Graphics Package Power Consumption for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    secondaryPresent = False
    printLogSpacer(' Power Consumption ')
    for device in deviceList:
        power_dict = getPower(device)
        power = 'N/A'
        if (
            power_dict['ret'] == rsmi_status_t.RSMI_STATUS_SUCCESS
            and power_dict['power_type'] != 'INVALID_POWER_TYPE'
        ):
            power = power_dict['power']
            printLog(
                device,
                power_dict['power_type'].title()
                + ' Graphics Package Power ('
                + power_dict['unit']
                + ')',
                power,
            )
        elif checkIfSecondaryDie(device):
            printLog(device, 'Average Graphics Package Power (W)', 'N/A (Secondary die)')
            secondaryPresent = True
        else:
            printErrLog(
                device,
                'Unable to get Average or Current Socket Graphics Package Power Consumption',
            )
    if secondaryPresent:
        printLog(
            None,
            '\n\t\tPrimary die (usually one above or below the secondary) shows total (primary + secondary) socket power information',
            None,
        )
    printLogSpacer()


def showPowerPlayTable(deviceList):
    """Display current GPU Memory clock frequencies and voltages for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    global PRINT_JSON
    if PRINT_JSON:
        return
    printLogSpacer(' GPU Memory clock frequencies and voltages ')
    odvf = rsmi_od_volt_freq_data_t()
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_od_volt_info_get(device, byref(odvf))
        if rsmi_ret_ok(ret, device, 'get_od_volt'):
            # TODO: Make this more dynamic and less hard-coded if possible
            printLog(device, 'OD_SCLK:', None)
            printLog(device, '0: %sMhz' % (int(odvf.curr_sclk_range.lower_bound / 1000000)), None)
            printLog(device, '1: %sMhz' % (int(odvf.curr_sclk_range.upper_bound / 1000000)), None)
            printLog(device, 'OD_MCLK:', None)
            printLog(device, '1: %sMhz' % (int(odvf.curr_mclk_range.upper_bound / 1000000)), None)
            if odvf.num_regions > 0:
                printLog(device, 'OD_VDDC_CURVE:', None)
                for position in range(3):
                    printLog(
                        device,
                        '%d: %sMhz %smV'
                        % (
                            position,
                            int(list(odvf.curve.vc_points)[position].frequency / 1000000),
                            int(list(odvf.curve.vc_points)[position].voltage),
                        ),
                        None,
                    )
            if (
                odvf.sclk_freq_limits.lower_bound > 0
                or odvf.sclk_freq_limits.upper_bound > 0
                or odvf.mclk_freq_limits.lower_bound > 0
                or odvf.mclk_freq_limits.upper_bound > 0
            ):
                printLog(device, 'OD_RANGE:', None)
            if odvf.sclk_freq_limits.lower_bound > 0 or odvf.sclk_freq_limits.upper_bound > 0:
                printLog(
                    device,
                    'SCLK:     %sMhz        %sMhz'
                    % (
                        int(odvf.sclk_freq_limits.lower_bound / 1000000),
                        int(odvf.sclk_freq_limits.upper_bound / 1000000),
                    ),
                    None,
                )
            if odvf.mclk_freq_limits.lower_bound > 0 or odvf.mclk_freq_limits.upper_bound > 0:
                printLog(
                    device,
                    'MCLK:     %sMhz        %sMhz'
                    % (
                        int(odvf.mclk_freq_limits.lower_bound / 1000000),
                        int(odvf.mclk_freq_limits.upper_bound / 1000000),
                    ),
                    None,
                )
            if odvf.num_regions > 0:
                for position in range(3):
                    printLog(
                        device,
                        'VDDC_CURVE_SCLK[%d]:     %sMhz'
                        % (position, int(list(odvf.curve.vc_points)[position].frequency / 1000000)),
                        None,
                    )
                    printLog(
                        device,
                        'VDDC_CURVE_VOLT[%d]:     %smV'
                        % (position, int(list(odvf.curve.vc_points)[position].voltage)),
                        None,
                    )
    printLogSpacer()


def showProduct(deviceList):
    """Show the requested product information for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Product Info ')
    for device in deviceList:
        # Only continue if GPU vendor is AMD
        if isAmdDevice(device):
            # TODO: Retrieve the SKU using 'rsmi_dev_sku_get' from the LIB
            # Device SKU is just the characters in between the two '-' in vbios_version
            vbios = getVbiosVersion(device, True)
            device_sku = 'N/A'
            if vbios.count('-') == 2 and len(str(vbios.split('-')[1])) > 1:
                device_sku = vbios.split('-')[1]

            printLog(device, 'Card Series', '\t\t' + str(getDeviceName(device)))
            # Retrieve device ID from DRM and KFD
            printLog(device, 'Card Model', str('\t\t' + getDRMDeviceId(device)))
            printLog(device, 'Card Vendor', '\t\t' + getVendor(device))
            printLog(device, 'Card SKU', '\t\t' + device_sku)
            printLog(device, 'Subsystem ID', str('\t' + getSubsystemId(device)))
            printLog(device, 'Device Rev', str('\t\t' + getRev(device)))
            printLog(device, 'Node ID', str('\t\t' + str(getNodeId(device))))
            printLog(device, 'GUID', str('\t\t' + str(getGUID(device))))
            printLog(device, 'GFX Version', str('\t\t' + getTargetGfxVersion(device)))

        else:
            vendor = getVendor(device)
            printLog(
                device,
                'Incompatible device.\n'
                'GPU[%s]\t\t: Expected vendor name: Advanced Micro Devices, Inc. [AMD/ATI]\n'
                'GPU[%s]\t\t: Actual vendor name' % (device, device),
                vendor,
            )
    printLogSpacer()


def showProfile(deviceList):
    """Display available Power Profiles for a list of devices.

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    global PRINT_JSON
    if PRINT_JSON:
        return
    printLogSpacer(' Show Power Profiles ')
    status = rsmi_power_profile_status_t()
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_power_profile_presets_get(device, 0, byref(status))
        if rsmi_ret_ok(ret, device, 'get_power_profiles', silent=False):
            binaryMaskString = str(format(status.available_profiles, '07b'))[::-1]
            bitMaskPosition = 0
            profileNumber = 0
            while bitMaskPosition < 7:
                if binaryMaskString[bitMaskPosition] == '1':
                    profileNumber = profileNumber + 1
                    if 2**bitMaskPosition == status.current:
                        printLog(
                            device,
                            '%d. Available power profile (#%d of 7)'
                            % (profileNumber, bitMaskPosition + 1),
                            profileString(2**bitMaskPosition) + '*',
                        )
                    else:
                        printLog(
                            device,
                            '%d. Available power profile (#%d of 7)'
                            % (profileNumber, bitMaskPosition + 1),
                            profileString(2**bitMaskPosition),
                        )
                bitMaskPosition = bitMaskPosition + 1
    printLogSpacer()


def showRange(deviceList, rangeType):
    """Show the range for either the sclk or voltage for the specified devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param rangeType: [sclk|voltage] Type of range to return
    """
    global RETCODE
    if rangeType not in {'sclk', 'mclk', 'voltage'}:
        printLog(None, 'Invalid range identifier %s' % (rangeType), None)
        RETCODE = 1
        return
    printLogSpacer(' Show Valid %s Range ' % (rangeType))
    odvf = rsmi_od_volt_freq_data_t()
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_od_volt_info_get(device, byref(odvf))
        if rsmi_ret_ok(ret, device, 'get_od_volt', silent=False):
            if rangeType == 'sclk':
                printLog(
                    device,
                    'Valid sclk range: %sMhz - %sMhz'
                    % (
                        int(odvf.curr_sclk_range.lower_bound / 1000000),
                        int(odvf.curr_sclk_range.upper_bound / 1000000),
                    ),
                    None,
                )
            if rangeType == 'mclk':
                printLog(
                    device,
                    'Valid mclk range: %sMhz - %sMhz'
                    % (
                        int(odvf.curr_mclk_range.lower_bound / 1000000),
                        int(odvf.curr_mclk_range.upper_bound / 1000000),
                    ),
                    None,
                )
            if rangeType == 'voltage':
                if odvf.num_regions == 0:
                    printErrLog(device, 'Voltage curve regions unsupported.')
                    continue
                num_regions = c_uint32(odvf.num_regions)
                regions = (rsmi_freq_volt_region_t * odvf.num_regions)()
                ret = rocmsmi.rsmi_dev_od_volt_curve_regions_get(
                    device,
                    byref(num_regions),
                    byref(regions),
                )
                if rsmi_ret_ok(ret, device, 'volt'):
                    for i in range(num_regions.value):
                        printLog(
                            device,
                            'Region %d: Valid voltage range: %smV - %smV'
                            % (
                                i,
                                regions[i].volt_range.lower_bound,
                                regions[i].volt_range.upper_bound,
                            ),
                            None,
                        )
                else:
                    printLog(device, 'Unable to display %s range' % (rangeType), None)
    printLogSpacer()


def showRasInfo(deviceList, rasType):
    """Show the requested RAS information for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param rasType: [$validRasBlocks] RAS counter to display (all if left empty)
    """
    state = rsmi_ras_err_state_t()
    if not rasType or 'all' in rasType:
        rasBlocks = rsmi_gpu_block_d.keys()
    else:
        for name in rasType:
            if name.upper() not in rsmi_gpu_block_d:
                rasType.remove(name)
                printErrLog(None, '%s is not a RAS block' % (name))

        rasBlocks = [block.upper() for block in rasType]

    printLogSpacer(' RAS Info ')
    for device in deviceList:
        data = []
        for block in rasBlocks:
            row = []
            ret = rocmsmi.rsmi_dev_ecc_status_get(device, rsmi_gpu_block_d[block], byref(state))
            if rsmi_ret_ok(ret, device, 'get_ecc_status_' + str(block), True):
                row.append(block)
                row.append(rsmi_ras_err_stale_machine[state.value].upper())
                # Now add the error count
                if (
                    rsmi_ras_err_stale_machine[state.value] != 'disabled'
                    or 'none'
                    or 'unknown error'
                ):
                    ec = rsmi_error_count_t()
                    ret = rocmsmi.rsmi_dev_ecc_count_get(device, rsmi_gpu_block_d[block], byref(ec))
                    if rsmi_ret_ok(ret, device, 'ecc err count', True):
                        row.append(ec.correctable_err)
                        row.append(ec.uncorrectable_err)
                data.append(row)
        printTableLog(
            ['         Block', '     Status  ', 'Correctable Error', 'Uncorrectable Error'],
            data,
            device,
            'RAS INFO',
        )
        # TODO: Use dynamic spacing for column widths
        printLogSpacer(None, '_')
    printLogSpacer()


def showRetiredPages(deviceList, retiredType='all'):
    """Show retired pages of a specified type for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param retiredType: Type of retired pages to show (default = all)
    """
    printLogSpacer(' Pages Info ')
    num_pages = c_uint32()
    records = rsmi_retired_page_record_t()

    for device in deviceList:
        data = []
        ret = rocmsmi.rsmi_dev_memory_reserved_pages_get(device, byref(num_pages), None)
        if rsmi_ret_ok(ret, device, 'ras'):
            records = (rsmi_retired_page_record_t * num_pages.value)()
        else:
            logging.debug('Unable to retrieve reserved page info')
            return

        ret = rocmsmi.rsmi_dev_memory_reserved_pages_get(device, byref(num_pages), byref(records))
        for rec in records:
            if memory_page_status_l[rec.status] == retiredType or retiredType == 'all':
                data.append(
                    (hex(rec.page_address), hex(rec.page_size), memory_page_status_l[rec.status]),
                )
        if data:
            printTableLog(
                ['    Page address', '   Page size', '    Status'],
                data,
                device,
                retiredType.upper() + ' PAGES INFO',
            )
    printLogSpacer()


def showSerialNumber(deviceList):
    """Display the serial number for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Serial Number ')
    for device in deviceList:
        sn = create_string_buffer(256)
        ret = rocmsmi.rsmi_dev_serial_number_get(device, sn, 256)
        try:
            sn.value.decode()
        except UnicodeDecodeError:
            printErrLog(
                device,
                'FRU Serial Number contains non-alphanumeric characters. FRU is likely corrupted',
            )
            continue

        if rsmi_ret_ok(ret, device, 'get_serial_number') and sn.value.decode():
            printLog(device, 'Serial Number', sn.value.decode())
        else:
            printLog(device, 'Serial Number', 'N/A')
    printLogSpacer()


def showUId(deviceList):
    """Display the unique device ID for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Unique ID ')
    for device in deviceList:
        dv_uid = c_uint64()
        ret = rocmsmi.rsmi_dev_unique_id_get(device, byref(dv_uid))
        if rsmi_ret_ok(ret, device, 'get_unique_id', True) and str(hex(dv_uid.value)):
            printLog(device, 'Unique ID', hex(dv_uid.value))
        else:
            printLog(device, 'Unique ID', 'N/A')
    printLogSpacer()


def showVbiosVersion(deviceList):
    """Display the VBIOS version for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' VBIOS ')
    for device in deviceList:
        printLog(device, 'VBIOS version', getVbiosVersion(device))
    printLogSpacer()


class _Getch:
    """
    Get a single character from standard input
    """

    def __init__(self):
        pass

    def __call__(self):
        import sys
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def showEvents(deviceList, eventTypes):
    """Display a blocking list of events for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    :param eventTypes: List of event type names (can be a single-item list)
    """
    printLogSpacer(' Show Events ')
    printLog(None, "press 'q' or 'ctrl + c' to quit", None)
    eventTypeList = []
    for event in eventTypes:  # Cleaning list from wrong values
        if event.replace(',', '').upper() in notification_type_names:
            eventTypeList.append(event.replace(',', '').upper())
        else:
            printErrLog(None, 'Ignoring unrecognized event type %s' % (event.replace(',', '')))
    if len(eventTypeList) == 0:
        eventTypeList = notification_type_names
        print2DArray([['DEVICE\t', 'TIME\t', 'TYPE\t', 'DESCRIPTION']])
        # Create a separate thread for each GPU
        for device in deviceList:
            try:
                _thread.start_new_thread(printEventList, (device, 1000, eventTypeList))
                time.sleep(0.25)
            except Exception as e:
                printErrLog(device, 'Unable to start new thread. %s' % (e))
                return
    while 1:  # Exit condition from user keyboard input of 'q' or 'ctrl + c'
        getch = _Getch()
        user_input = getch()
        # Catch user input for q or Ctrl + c
        if user_input == 'q' or user_input == '\x03':
            for device in deviceList:
                ret = rocmsmi.rsmi_event_notification_stop(device)
                if not rsmi_ret_ok(ret, device, 'stop_event_notification'):
                    printErrLog(device, 'Unable to end event notifications.')
            print('\r')
            break


def printTempGraph(deviceList, delay, temp_type):
    # deviceList must be in ascending order
    deviceList.sort()
    devices = 0
    # Print an empty line for each device
    for device in deviceList:
        devices = devices + 1
    for i in range(devices):
        printEmptyLine()
    originalTerminalWidth = os.get_terminal_size()[0]
    while 1:  # Exit condition from user keyboard input of 'q' or 'ctrl + c'
        terminalWidth = os.get_terminal_size()[0]
        printStrings = list()
        for device in deviceList:
            temp = getTemp(device, temp_type)
            if temp == 'N/A':
                percentage = 0
            else:
                percentage = temp
            if percentage >= 100:
                percentage = 100
            if percentage < 0:
                percentage = 0
            # Get available space based on terminal width
            availableSpace = 0
            if terminalWidth >= 20:
                availableSpace = terminalWidth - 20
            # Get color based on percentage, with a non-linear scaling
            color = getGraphColor(3.16 * (percentage**1.5) ** (1 / 2))
            # Get graph length based on percentage and available space
            padding = (percentage / float(100)) * availableSpace
            if padding > availableSpace:
                padding = availableSpace
            paddingSpace = color[-1]
            for i in range(int(padding)):
                paddingSpace += paddingSpace[-1]
            remainder = 0
            if availableSpace >= padding:
                remainder = availableSpace + 1 - padding
            remainderSpace = ' ' * int(remainder)
            # TODO: Allow terminal size to be decreased
            if terminalWidth < originalTerminalWidth:
                print('Terminal size cannot be decreased.\n\r')
                return
            if type(temp) == str:
                tempString = temp
            else:
                tempString = str(int(temp))
            # Two spare Spaces
            tempString = (tempString + '°C').ljust(5)
            printStrings.append(
                '\033[2;30;47mGPU[%d] Temp %s|%s%s\x1b[0m%s'
                % (device, tempString, color, paddingSpace[1:], remainderSpace),
            )
            originalTerminalWidth = terminalWidth
            time.sleep(delay / 1000)

        if terminalWidth >= 20:
            # go up and prepare to rewrite the lines
            for i in printStrings:
                print('\033[A', end='\r')
            # print all strings
            for i in printStrings:
                print(i, end='\r\n')


def getGraphColor(percentage):
    # Text / Background color mixing (Tested on PuTTY)
    colors = [
        '\033[2;35;45m',
        '\033[2;34;45m',
        '\033[2;35;44m',
        '\033[2;34;44m',
        '\033[2;36;44m',
        '\033[2;34;46m',
        '\033[2;36;46m',
        '\033[2;32;46m',
        '\033[2;36;42m',
        '\033[2;32;42m',
        '\033[2;33;42m',
        '\033[2;32;43m',
        '\033[2;33;43m',
        '\033[2;31;43m',
        '\033[2;33;41m',
        '\033[2;31;41m',
    ]
    characters = [' ', '░', '░', '▒', '▒', '░']
    # Ensure percentage is in range and rounded
    if percentage > 99:
        percentage = 99
    if percentage < 0:
        percentage = 0
    percentage = round(percentage, 0)
    # There are a total of 16 distinct colors, with 2 special ascii characters per
    # color, for a total of 16*2=32 distinct colors for a gradient.
    # Therefore every 100/32=3.125 percent the color gradient will change
    stepSize = (100 / len(colors)) / 2
    characterIndex = int((percentage % (len(characters) * stepSize)) / stepSize)
    colorIndex = int(percentage / (stepSize * 2))
    returnStr = colors[colorIndex] + characters[characterIndex]
    return returnStr


def showTempGraph(deviceList):
    deviceList.sort()
    temp_type = getTemperatureLabel(deviceList)
    printLogSpacer(' Temperature Graph ' + temp_type.capitalize() + ' ')
    # Start a thread for constantly printing
    try:
        # Create a thread (call print function, devices, delay in ms)
        _thread.start_new_thread(printTempGraph, (deviceList, 150, temp_type))
    except Exception as e:
        printErrLog(device, 'Unable to start new thread. %s' % (e))
    # Catch user input for program termination
    while 1:  # Exit condition from user keyboard input of 'q' or 'ctrl + c'
        getch = _Getch()
        user_input = getch()
        # Catch user input for q or Ctrl + c
        if user_input == 'q' or user_input == '\x03':
            break
    # Reset color to default before exit
    print('\033[A\x1b[0m\r')
    printLogSpacer()


def showDriverVersion(deviceList, component):
    """Display the software version for the specified component

    :param deviceList: List of DRM devices (can be a single-item list)
    :param component: Component (currently only driver)
    """
    printLogSpacer(' Version of System Component ')
    printSysLog(component_str(component) + ' version', getVersion(deviceList, component))
    printLogSpacer()


def showVoltage(deviceList):
    """Display the current voltage (in millivolts) for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Current voltage ')
    for device in deviceList:
        vtype = rsmi_voltage_type_t(0)
        met = rsmi_voltage_metric_t(0)
        voltage = c_uint64()
        ret = rocmsmi.rsmi_dev_volt_metric_get(device, vtype, met, byref(voltage))
        if rsmi_ret_ok(ret, device, 'get_volt_metric') and str(voltage.value):
            printLog(device, 'Voltage (mV)', str(voltage.value))
        else:
            logging.debug('GPU voltage not supported')
    printLogSpacer()


def showVoltageCurve(deviceList):
    """Show the voltage curve points for the specified devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Voltage Curve Points ')
    odvf = rsmi_od_volt_freq_data_t()
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_od_volt_info_get(device, byref(odvf))
        if rsmi_ret_ok(ret, device, 'get_od_volt_info', silent=False) and odvf.num_regions > 0:
            for position in range(3):
                printLog(
                    device,
                    'Voltage point %d: %sMhz %smV'
                    % (
                        position,
                        int(list(odvf.curve.vc_points)[position].frequency / 1000000),
                        int(list(odvf.curve.vc_points)[position].voltage),
                    ),
                    None,
                )
        else:
            printErrLog(device, 'Voltage curve Points unsupported.')
    printLogSpacer()


def showXgmiErr(deviceList):
    """Display the XGMI Error status

    This reads the XGMI error file, and interprets the return value from the sysfs file

    :param deviceList: Show XGMI error state for these devices
    """
    printLogSpacer('XGMI Error status')
    xe = rsmi_xgmi_status_t()
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_xgmi_error_status(device, byref(xe))
        if rsmi_ret_ok(ret, device, 'xgmi status'):
            desc = ''
            if xe.value is None:
                continue
            else:
                err = int(xe.value)
            if err == 0:
                desc = 'No errors detected since last read'
            elif err == 1:
                desc = 'Single error detected since last read'
            elif err == 2:
                desc = 'Multiple errors detected since last read'
            else:
                printErrLog(device, 'Invalid return value from xgmi_error')
                continue
            if PRINT_JSON is True:
                printLog(device, 'XGMI Error count', err)
            else:
                printLog(device, 'XGMI Error count', f'{err} ({desc})')
    printLogSpacer()


def showAccessibleTopology(deviceList):
    """Display the HW Topology Information based on link accessibility

    This reads the HW Topology file and displays the matrix for the nodes

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    devices_ind = range(len(deviceList))
    accessible = c_bool()
    gpu_links_type = [[0 for x in devices_ind] for y in devices_ind]
    printLogSpacer(' Link accessibility between two GPUs ')
    for srcdevice in deviceList:
        for destdevice in deviceList:
            ret = rocmsmi.rsmi_is_P2P_accessible(srcdevice, destdevice, byref(accessible))
            if rsmi_ret_ok(ret, metric='is_P2P_accessible'):
                gpu_links_type[srcdevice][destdevice] = accessible.value
            else:
                printErrLog(
                    srcdevice,
                    'Cannot read link accessibility: Unsupported on this machine',
                )
    if PRINT_JSON:
        formatMatrixToJSON(
            deviceList,
            gpu_links_type,
            '(Topology) Link accessibility between DRM devices {} and {}',
        )
        return

    printTableRow(None, '      ')
    for row in deviceList:
        tmp = 'GPU%d' % row
        printTableRow('%-12s', tmp)
    printEmptyLine()
    for gpu1 in deviceList:
        tmp = 'GPU%d' % gpu1
        printTableRow('%-6s', tmp)
        for gpu2 in deviceList:
            printTableRow('%-12s', gpu_links_type[gpu1][gpu2])
        printEmptyLine()


def showWeightTopology(deviceList):
    """Display the HW Topology Information based on weights

    This reads the HW Topology file and displays the matrix for the nodes

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    global PRINT_JSON
    devices_ind = range(len(deviceList))
    gpu_links_weight = [[0 for x in devices_ind] for y in devices_ind]
    printLogSpacer(' Weight between two GPUs ')
    for srcdevice in deviceList:
        for destdevice in deviceList:
            if srcdevice == destdevice:
                gpu_links_weight[srcdevice][destdevice] = 0
                continue
            weight = c_uint64()
            ret = rocmsmi.rsmi_topo_get_link_weight(srcdevice, destdevice, byref(weight))
            if rsmi_ret_ok(ret, metric='get_link_weight_topology'):
                gpu_links_weight[srcdevice][destdevice] = weight
            else:
                printErrLog(srcdevice, 'Cannot read Link Weight: Not supported on this machine')
                gpu_links_weight[srcdevice][destdevice] = None

    if PRINT_JSON:
        formatMatrixToJSON(
            deviceList,
            gpu_links_weight,
            '(Topology) Weight between DRM devices {} and {}',
        )
        return

    printTableRow(None, '      ')
    for row in deviceList:
        tmp = 'GPU%d' % row
        printTableRow('%-12s', tmp)
    printEmptyLine()
    for gpu1 in deviceList:
        tmp = 'GPU%d' % gpu1
        printTableRow('%-6s', tmp)
        for gpu2 in deviceList:
            if gpu1 == gpu2:
                printTableRow('%-12s', '0')
            elif gpu_links_weight[gpu1][gpu2] == None:
                printTableRow('%-12s', 'N/A')
            else:
                printTableRow('%-12s', gpu_links_weight[gpu1][gpu2].value)
        printEmptyLine()


def showHopsTopology(deviceList):
    """Display the HW Topology Information based on number of hops

    This reads the HW Topology file and displays the matrix for the nodes

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    linktype = c_char_p()
    devices_ind = range(len(deviceList))
    gpu_links_hops = [[0 for x in devices_ind] for y in devices_ind]
    printLogSpacer(' Hops between two GPUs ')
    for srcdevice in deviceList:
        for destdevice in deviceList:
            if srcdevice == destdevice:
                gpu_links_hops[srcdevice][destdevice] = '0'
                continue
            hops = c_uint64()
            ret = rocmsmi.rsmi_topo_get_link_type(
                srcdevice,
                destdevice,
                byref(hops),
                byref(linktype),
            )
            if rsmi_ret_ok(ret, metric='get_link_type_topology'):
                gpu_links_hops[srcdevice][destdevice] = hops
            else:
                printErrLog(srcdevice, 'Cannot read Link Hops: Not supported on this machine')
                gpu_links_hops[srcdevice][destdevice] = None

    if PRINT_JSON:
        formatMatrixToJSON(
            deviceList,
            gpu_links_hops,
            '(Topology) Hops between DRM devices {} and {}',
        )
        return

    printTableRow(None, '      ')
    for row in deviceList:
        tmp = 'GPU%d' % row
        printTableRow('%-12s', tmp)
    printEmptyLine()
    for gpu1 in deviceList:
        tmp = 'GPU%d' % gpu1
        printTableRow('%-6s', tmp)
        for gpu2 in deviceList:
            if gpu1 == gpu2:
                printTableRow('%-12s', '0')
            elif gpu_links_hops[gpu1][gpu2] == None:
                printTableRow('%-12s', 'N/A')
            else:
                printTableRow('%-12s', gpu_links_hops[gpu1][gpu2].value)
        printEmptyLine()


def showTypeTopology(deviceList):
    """Display the HW Topology Information based on link type

    This reads the HW Topology file and displays the matrix for the nodes

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    devices_ind = range(len(deviceList))
    hops = c_uint64()
    linktype = c_uint64()
    gpu_links_type = [[0 for x in devices_ind] for y in devices_ind]
    printLogSpacer(' Link Type between two GPUs ')
    for srcdevice in deviceList:
        for destdevice in deviceList:
            if srcdevice == destdevice:
                gpu_links_type[srcdevice][destdevice] = '0'
                continue
            ret = rocmsmi.rsmi_topo_get_link_type(
                srcdevice,
                destdevice,
                byref(hops),
                byref(linktype),
            )
            if rsmi_ret_ok(ret, metric='get_link_topology_type'):
                if linktype.value == 1:
                    gpu_links_type[srcdevice][destdevice] = 'PCIE'
                elif linktype.value == 2:
                    gpu_links_type[srcdevice][destdevice] = 'XGMI'
                else:
                    gpu_links_type[srcdevice][destdevice] = 'XXXX'
            else:
                printErrLog(srcdevice, 'Cannot read Link Type: Not supported on this machine')
                gpu_links_type[srcdevice][destdevice] = 'XXXX'

    if PRINT_JSON:
        formatMatrixToJSON(
            deviceList,
            gpu_links_type,
            '(Topology) Link type between DRM devices {} and {}',
        )
        return

    printTableRow(None, '      ')
    for row in deviceList:
        tmp = 'GPU%d' % row
        printTableRow('%-12s', tmp)
    printEmptyLine()
    for gpu1 in deviceList:
        tmp = 'GPU%d' % gpu1
        printTableRow('%-6s', tmp)
        for gpu2 in deviceList:
            if gpu1 == gpu2:
                printTableRow('%-12s', '0')
            else:
                printTableRow('%-12s', gpu_links_type[gpu1][gpu2])
        printEmptyLine()


def showNumaTopology(deviceList):
    """Display the HW Topology Information for numa nodes

    This reads the HW Topology file and display the matrix for the nodes

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    printLogSpacer(' Numa Nodes ')
    numa_numbers = c_int32()
    for device in deviceList:
        ret = rocmsmi.rsmi_topo_get_numa_node_number(device, byref(numa_numbers))
        if rsmi_ret_ok(ret, device, 'get_numa_node_number'):
            printLog(device, '(Topology) Numa Node', numa_numbers.value)
        else:
            printErrLog(device, 'Cannot read Numa Node')

        ret = rocmsmi.rsmi_topo_numa_affinity_get(device, byref(numa_numbers))
        if rsmi_ret_ok(ret, metric='get_numa_affinity_topology'):
            printLog(device, '(Topology) Numa Affinity', numa_numbers.value)
        else:
            printErrLog(device, 'Cannot read Numa Affinity')


def showHwTopology(deviceList):
    """Display the HW Topology Information based on weight/hops/type

    This reads the HW Topology file and displays the matrix for the nodes

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    showWeightTopology(deviceList)
    printEmptyLine()
    showHopsTopology(deviceList)
    printEmptyLine()
    showTypeTopology(deviceList)
    printEmptyLine()
    showNumaTopology(deviceList)


def showNodesBw(deviceList):
    """Display max and min bandwidth between nodes.
    Currently supports XGMI only.
    This reads the HW Topology file and displays the matrix for the nodes
    :param deviceList: List of DRM devices (can be a single-item list)
    """
    devices_ind = range(len(deviceList))
    minBW = c_uint32()
    maxBW = c_uint32()
    hops = c_uint64()
    linktype = c_uint64()
    silent = False
    nonXgmi = False
    gpu_links_type = [[0 for x in devices_ind] for y in devices_ind]
    printLogSpacer(' Bandwidth ')
    for srcdevice in deviceList:
        for destdevice in deviceList:
            if srcdevice != destdevice:
                ret = rocmsmi.rsmi_minmax_bandwidth_get(
                    srcdevice,
                    destdevice,
                    byref(minBW),
                    byref(maxBW),
                )
                # verify that link type is xgmi
                ret2 = rocmsmi.rsmi_topo_get_link_type(
                    srcdevice,
                    destdevice,
                    byref(hops),
                    byref(linktype),
                )
                if rsmi_ret_ok(
                    ret2,
                    f' {srcdevice} to {destdevice}',
                    'get_link_topology_type',
                    True,
                ):
                    if linktype.value != 2:
                        nonXgmi = True
                        silent = True
                        gpu_links_type[srcdevice][destdevice] = 'N/A'

                if rsmi_ret_ok(
                    ret,
                    f' {srcdevice}  to {destdevice}',
                    'get_link_topology_type',
                    silent,
                ):
                    gpu_links_type[srcdevice][destdevice] = f'{minBW.value}-{maxBW.value}'
            else:
                gpu_links_type[srcdevice][destdevice] = 'N/A'
    if PRINT_JSON:
        # TODO
        return
    printTableRow(None, '      ')
    for row in deviceList:
        tmp = 'GPU%d' % row
        printTableRow('%-12s', tmp)
    printEmptyLine()
    for gpu1 in deviceList:
        tmp = 'GPU%d' % gpu1
        printTableRow('%-6s', tmp)
        for gpu2 in deviceList:
            printTableRow('%-12s', gpu_links_type[gpu1][gpu2])
        printEmptyLine()
    printLog(None, 'Format: min-max; Units: mps', None)
    printLog(None, '"0-0" min-max bandwidth indicates devices are not connected directly', None)
    if nonXgmi:
        printLog(None, 'Non-xGMI links detected and is currently not supported', None)


def showComputePartition(deviceList):
    """Returns the current compute partitioning for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    currentComputePartition = create_string_buffer(256)
    printLogSpacer(' Current Compute Partition ')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_compute_partition_get(device, currentComputePartition, 256)
        if (
            rsmi_ret_ok(ret, device, 'get_compute_partition', silent=True)
            and currentComputePartition.value.decode()
        ):
            printLog(device, 'Compute Partition', currentComputePartition.value.decode())
        elif ret == rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED:
            printLog(device, 'Not supported on the given system', None)
        else:
            rsmi_ret_ok(ret, device, 'get_compute_partition')
            printErrLog(
                device,
                'Failed to retrieve compute partition, even though device supports it.',
            )
    printLogSpacer()


def showMemoryPartition(deviceList):
    """Returns the current memory partition for a list of devices

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    memoryPartition = create_string_buffer(256)
    printLogSpacer(' Current Memory Partition ')
    for device in deviceList:
        ret = rocmsmi.rsmi_dev_memory_partition_get(device, memoryPartition, 256)
        if (
            rsmi_ret_ok(ret, device, 'get_memory_partition', silent=True)
            and memoryPartition.value.decode()
        ):
            printLog(device, 'Memory Partition', memoryPartition.value.decode())
        elif ret == rsmi_status_t.RSMI_STATUS_NOT_SUPPORTED:
            printLog(device, 'Not supported on the given system', None)
        else:
            rsmi_ret_ok(ret, device, 'get_memory_partition')
            printErrLog(
                device,
                'Failed to retrieve current memory partition, even though device supports it.',
            )
    printLogSpacer()


def checkAmdGpus(deviceList):
    """Check if there are any AMD GPUs being queried,
    return False if there are none

    :param deviceList: List of DRM devices (can be a single-item list)
    """
    for device in deviceList:
        if isAmdDevice(device):
            return True
    return False


def component_str(component):
    """Returns the component String value

    :param component: Component (currently only driver)
    """
    switcher = {
        0: 'Driver',
    }
    return switcher.get(component, 'UNKNOWN')


def confirmOutOfSpecWarning(autoRespond):
    """Print the warning for running outside of specification and prompt user to accept the terms.

    :param autoRespond: Response to automatically provide for all prompts
    """
    print(
        """
          ******WARNING******\n
          Operating your AMD GPU outside of official AMD specifications or outside of
          factory settings, including but not limited to the conducting of overclocking,
          over-volting or under-volting (including use of this interface software,
          even if such software has been directly or indirectly provided by AMD or otherwise
          affiliated in any way with AMD), may cause damage to your AMD GPU, system components
          and/or result in system failure, as well as cause other problems.
          DAMAGES CAUSED BY USE OF YOUR AMD GPU OUTSIDE OF OFFICIAL AMD SPECIFICATIONS OR
          OUTSIDE OF FACTORY SETTINGS ARE NOT COVERED UNDER ANY AMD PRODUCT WARRANTY AND
          MAY NOT BE COVERED BY YOUR BOARD OR SYSTEM MANUFACTURER'S WARRANTY.
          Please use this utility with caution.
          """,
    )
    if not autoRespond:
        user_input = input('Do you accept these terms? [y/N] ')
    else:
        user_input = autoRespond
    if user_input in ['Yes', 'yes', 'y', 'Y', 'YES']:
        return
    else:
        sys.exit('Confirmation not given. Exiting without setting value')


def doesDeviceExist(device):
    """Check whether the specified device exists

    :param device: DRM device identifier
    """
    availableDevices = listDevices()
    filePath = '/sys/kernel/debug/dri/%d/' % (int(device))
    if device in availableDevices or os.path.exists(filePath):
        return True
    return False


def initializeRsmi():
    """initializes rocmsmi if the amdgpu driver is initialized"""
    global rocmsmi
    # Initialize rsmiBindings
    rocmsmi = initRsmiBindings(silent=PRINT_JSON)
    # Check if amdgpu is initialized before initializing rsmi
    if driverInitialized() is True:
        ret_init = rocmsmi.rsmi_init(0)
        if ret_init != 0:
            logging.error('ROCm SMI returned %s (the expected value is 0)', ret_init)
            exit(ret_init)
    else:
        logging.error('Driver not initialized (amdgpu not found in modules)')
        exit(0)


def isAmdDevice(device):
    """Return whether the specified device is an AMD device or not

    :param device: DRM device identifier
    """
    vendorID = c_uint16()
    # Retrieve card vendor
    ret = rocmsmi.rsmi_dev_vendor_id_get(device, byref(vendorID))
    # Only continue if GPU vendor is AMD, which is 1002
    if ret == rsmi_status_t.RSMI_STATUS_SUCCESS and str(hex(vendorID.value)) == '0x1002':
        return True
    return False


def listDevices():
    """Returns a list of GPU devices"""
    global rocmsmi
    numberOfDevices = c_uint32(0)
    ret = rocmsmi.rsmi_num_monitor_devices(byref(numberOfDevices))
    if rsmi_ret_ok(ret, metric='get_num_monitor_devices'):
        deviceList = list(range(numberOfDevices.value))
        return deviceList
    else:
        exit(ret)


def load(savefilepath, autoRespond):
    """Load clock frequencies and fan speeds from a specified file.

    :param savefilepath: Path to the save file
    :param autoRespond: Response to automatically provide for all prompts
    """
    printLogSpacer(' Load Settings ')
    if not os.path.isfile(savefilepath):
        printLog(None, 'No settings file found at %s' % (savefilepath), None)
        printLogSpacer()
        sys.exit()
    with open(savefilepath) as savefile:
        jsonData = json.loads(savefile.read())
        for device, values in jsonData.items():
            if values['vJson'] != CLOCK_JSON_VERSION:
                printLog(
                    None,
                    'Unable to load legacy clock file - file v%s != current v%s'
                    % (str(values['vJson']), str(CLOCK_JSON_VERSION)),
                    None,
                )
                break
            device = int(device[4:])
            if values['fan']:
                setFanSpeed([device], values['fan'])
            if values['overdrivesclk']:
                setClockOverDrive([device], 'sclk', values['overdrivesclk'], autoRespond)
            if values['overdrivemclk']:
                setClockOverDrive([device], 'mclk', values['overdrivemclk'], autoRespond)
            for clk in validClockNames:
                if clk in values['clocks']:
                    setClocks([device], clk, values['clocks'][clk])
            if values['profile']:
                setProfile([device], values['profile'])
            # Set Perf level last, since setting OverDrive sets the Performance level
            # to manual, and Profiles only work when the Performance level is auto
            if values['perflevel'] != -1:
                setPerformanceLevel([device], values['perflevel'])
            printLog(device, 'Successfully loaded values from ' + savefilepath, None)
    printLogSpacer()


def padHexValue(value, length):
    """Pad a hexadecimal value with a given length of zeros

    :param value: A hexadecimal value to be padded with zeros
    :param length: Number of zeros to pad the hexadecimal value
    """
    # Ensure value entered meets the minimum length and is hexadecimal
    if (
        len(value) > 2
        and length > 1
        and value[:2].lower() == '0x'
        and all(c in '0123456789abcdefABCDEF' for c in value[2:])
    ):
        # Pad with zeros after '0x' prefix
        return '0x' + value[2:].zfill(length)
    return value


def profileString(profile):
    dictionary = {
        1: 'CUSTOM',
        2: 'VIDEO',
        4: 'POWER SAVING',
        8: 'COMPUTE',
        16: 'VR',
        32: '3D FULL SCREEN',
        64: 'BOOTUP DEFAULT',
    }
    # TODO: We should dynamically generate this to avoid hardcoding
    if str(profile).isnumeric() and int(profile) in dictionary:
        return dictionary.get(int(profile))
    elif not str(profile).isnumeric() and str(profile) in dictionary.values():
        return list(dictionary.keys())[list(dictionary.values()).index(str(profile))]
    return 'UNKNOWN'


def relaunchAsSudo():
    """Relaunch the SMI as sudo

    To use rocm_smi_lib functions that write to sysfs, the SMI requires root access
    Use execvp to relaunch the script with sudo privileges
    """
    if os.geteuid() != 0:
        os.execvp('sudo', ['sudo'] + sys.argv)
        # keeping below, if we want to run sudo with user's env variables
        # os.execvp('sudo', ['sudo', '-E'] + sys.argv)


def rsmi_ret_ok(my_ret, device=None, metric=None, silent=False):
    """Returns true if RSMI call status is 0 (success)

        If status is not 0, error logs are written to the debug log and false is returned

    :param device: DRM device identifier
    :param my_ret: Return of RSMI call (rocm_smi_lib API)
    :param metric: Parameter of GPU currently being analyzed
    :param silent: Echo verbose error response.
        True silences err output, False does not silence err output (default).
    """
    global RETCODE
    global PRINT_JSON
    if my_ret != rsmi_status_t.RSMI_STATUS_SUCCESS:
        err_str = c_char_p()
        rocmsmi.rsmi_status_string(my_ret, byref(err_str))
        # leaving the commented out prints/logs to help identify errors in the future
        # print("error string = " + str(err_str))
        # print("error string (w/ decode)= " + str(err_str.value.decode()))
        returnString = ''
        if device is not None:
            returnString += f'{my_ret} GPU[{device}]:'
        if metric is not None:
            returnString += ' %s: ' % (metric)
        else:
            metric = ''
        if err_str.value is not None:
            returnString += '%s\t' % (err_str.value.decode())
        if not PRINT_JSON:
            # logging.debug('%s', returnString)
            if not silent:
                logging.debug('%s', returnString)
                if my_ret in rsmi_status_verbose_err_out:
                    printLog(device, metric + ', ' + rsmi_status_verbose_err_out[my_ret], None)
        RETCODE = my_ret
        return False
    return True


def save(deviceList, savefilepath):
    """Save clock frequencies and fan speeds for a list of devices to a specified file path.

    :param deviceList: List of DRM devices (can be a single-item list)
    :param savefilepath: Path to use to create the save file
    """
    perfLevels = {}
    clocks = {}
    fanSpeeds = {}
    overDriveGpu = {}
    overDriveGpuMem = {}
    profiles = {}
    jsonData = {}
    printLogSpacer(' Save Settings ')
    if os.path.isfile(savefilepath):
        printLog(None, '%s already exists. Settings not saved' % (savefilepath), None)
        printLogSpacer()
        sys.exit()
    for device in deviceList:
        if getPerfLevel(device) != -1:
            perfLevels[device] = str(getPerfLevel(device)).lower()
        else:
            perfLevels[device] = 'Unsupported'
        freq = rsmi_frequencies_t()
        for clk_type in sorted(rsmi_clk_names_dict):
            clocks[device] = clocks.get(device, {})
            ret = rocmsmi.rsmi_dev_gpu_clk_freq_get(
                device,
                rsmi_clk_names_dict[clk_type],
                byref(freq),
            )
            if rsmi_ret_ok(ret, device, 'get_gpu_clk_freq_' + str(clk_type), True):
                clocks[device][clk_type] = str(freq.current)
            else:
                clocks[device][clk_type] = '0'
        fanSpeeds[device] = getFanSpeed(device)[1]
        od = c_uint32()
        ret = rocmsmi.rsmi_dev_overdrive_level_get(device, byref(od))
        if rsmi_ret_ok(ret, device, 'get_overdrive_level'):
            overDriveGpu[device] = str(od.value)
        else:
            overDriveGpu[device] = '0'
        # GPU memory Overdrive is legacy
        overDriveGpuMem[device] = '0'
        status = rsmi_power_profile_status_t()
        ret = rocmsmi.rsmi_dev_power_profile_presets_get(device, 0, byref(status))
        if rsmi_ret_ok(ret, device, 'get_profile_presets'):
            profiles[device] = str(str(bin(status.current))[2:][::-1].index('1') + 1)
        else:
            profiles[device] = 'UNKNOWN'
        jsonData['card%d' % (device)] = {
            'vJson': CLOCK_JSON_VERSION,
            'clocks': clocks[device],
            'fan': fanSpeeds[device],
            'overdrivesclk': overDriveGpu[device],
            'overdrivemclk': overDriveGpuMem[device],
            'profile': profiles[device],
            'perflevel': perfLevels[device],
        }
    printLog(None, 'Current settings successfully saved to', savefilepath)
    with open(savefilepath, 'w') as savefile:
        json.dump(jsonData, savefile, ensure_ascii=True)
    printLogSpacer()


# The code below is for when this script is run as an executable instead of when imported as a module
def isConciseInfoRequested(args):
    is_concise_req = (
        len(sys.argv) == 1
        or len(sys.argv) == 2
        and (args.alldevices or (args.json or args.csv))
        or len(sys.argv) == 3
        and (args.alldevices and (args.json or args.csv))
    )
    return is_concise_req


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AMD ROCm System Management Interface  |  ROCM-SMI version: %s' % __version__,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=90, width=120),
    )
    groupVersion = parser.add_argument_group()
    groupDev = parser.add_argument_group()
    groupDisplayOpt = parser.add_argument_group('Display Options')
    groupDisplayTop = parser.add_argument_group('Topology')
    groupDisplayPages = parser.add_argument_group('Pages information')
    groupDisplayHw = parser.add_argument_group('Hardware-related information')
    groupDisplay = parser.add_argument_group('Software-related/controlled information')
    groupAction = parser.add_argument_group('Set options')
    groupActionReset = parser.add_argument_group('Reset options')
    groupActionGpuReset = parser.add_mutually_exclusive_group()
    groupFile = parser.add_mutually_exclusive_group()
    groupResponse = parser.add_argument_group('Auto-response options')
    groupActionOutput = parser.add_argument_group('Output options')

    groupVersion.add_argument(
        '-V',
        '--version',
        help='Show version information',
        action='store_true',
    )
    groupDev.add_argument(
        '-d',
        '--device',
        help='Execute command on specified device',
        type=int,
        nargs='+',
    )
    groupDisplayOpt.add_argument(
        '--alldevices',
        action='store_true',
    )  # ------------- function deprecated, no help menu
    groupDisplayOpt.add_argument('--showhw', help='Show Hardware details', action='store_true')
    groupDisplayOpt.add_argument(
        '-a',
        '--showallinfo',
        help='Show Temperature, Fan and Clock values',
        action='store_true',
    )
    groupDisplayTop.add_argument('-i', '--showid', help='Show DEVICE IDs', action='store_true')
    groupDisplayTop.add_argument(
        '-v',
        '--showvbios',
        help='Show VBIOS version',
        action='store_true',
    )
    groupDisplayTop.add_argument(
        '-e',
        '--showevents',
        help='Show event list',
        metavar='EVENT',
        type=str,
        nargs='*',
    )
    groupDisplayTop.add_argument(
        '--showdriverversion',
        help='Show kernel driver version',
        action='store_true',
    )
    groupDisplayTop.add_argument(
        '--showtempgraph',
        help='Show Temperature Graph',
        action='store_true',
    )
    groupDisplayTop.add_argument(
        '--showfwinfo',
        help='Show FW information',
        metavar='BLOCK',
        type=str,
        nargs='*',
    )
    groupDisplayTop.add_argument('--showmclkrange', help='Show mclk range', action='store_true')
    groupDisplayTop.add_argument(
        '--showmemvendor',
        help='Show GPU memory vendor',
        action='store_true',
    )
    groupDisplayTop.add_argument('--showsclkrange', help='Show sclk range', action='store_true')
    groupDisplayTop.add_argument(
        '--showproductname',
        help='Show product details',
        action='store_true',
    )
    groupDisplayTop.add_argument(
        '--showserial',
        help="Show GPU's Serial Number",
        action='store_true',
    )
    groupDisplayTop.add_argument('--showuniqueid', help="Show GPU's Unique ID", action='store_true')
    groupDisplayTop.add_argument(
        '--showvoltagerange',
        help='Show voltage range',
        action='store_true',
    )
    groupDisplayTop.add_argument('--showbus', help='Show PCI bus number', action='store_true')
    groupDisplayPages.add_argument(
        '--showpagesinfo',
        help='Show retired, pending and unreservable pages',
        action='store_true',
    )
    groupDisplayPages.add_argument(
        '--showpendingpages',
        help='Show pending retired pages',
        action='store_true',
    )
    groupDisplayPages.add_argument(
        '--showretiredpages',
        help='Show retired pages',
        action='store_true',
    )
    groupDisplayPages.add_argument(
        '--showunreservablepages',
        help='Show unreservable pages',
        action='store_true',
    )
    groupDisplayHw.add_argument(
        '-f',
        '--showfan',
        help='Show current fan speed',
        action='store_true',
    )
    groupDisplayHw.add_argument(
        '-P',
        '--showpower',
        help='Show current Average Graphics Package Power Consumption',
        action='store_true',
    )
    groupDisplayHw.add_argument(
        '-t',
        '--showtemp',
        help='Show current temperature',
        action='store_true',
    )
    groupDisplayHw.add_argument('-u', '--showuse', help='Show current GPU use', action='store_true')
    groupDisplayHw.add_argument(
        '--showmemuse',
        help='Show current GPU memory used',
        action='store_true',
    )
    groupDisplayHw.add_argument(
        '--showvoltage',
        help='Show current GPU voltage',
        action='store_true',
    )
    groupDisplay.add_argument('-b', '--showbw', help='Show estimated PCIe use', action='store_true')
    groupDisplay.add_argument(
        '-c',
        '--showclocks',
        help='Show current clock frequencies',
        action='store_true',
    )
    groupDisplay.add_argument(
        '-g',
        '--showgpuclocks',
        help='Show current GPU clock frequencies',
        action='store_true',
    )
    groupDisplay.add_argument(
        '-l',
        '--showprofile',
        help='Show Compute Profile attributes',
        action='store_true',
    )
    groupDisplay.add_argument(
        '-M',
        '--showmaxpower',
        help='Show maximum graphics package power this GPU will consume',
        action='store_true',
    )
    groupDisplay.add_argument(
        '-m',
        '--showmemoverdrive',
        help='Show current GPU Memory Clock OverDrive level',
        action='store_true',
    )
    groupDisplay.add_argument(
        '-o',
        '--showoverdrive',
        help='Show current GPU Clock OverDrive level',
        action='store_true',
    )
    groupDisplay.add_argument(
        '-p',
        '--showperflevel',
        help='Show current DPM Performance Level',
        action='store_true',
    )
    groupDisplay.add_argument(
        '-S',
        '--showclkvolt',
        help='Show supported GPU and Memory Clocks and Voltages',
        action='store_true',
    )
    groupDisplay.add_argument(
        '-s',
        '--showclkfrq',
        help='Show supported GPU and Memory Clock',
        action='store_true',
    )
    groupDisplay.add_argument(
        '--showmeminfo',
        help='Show Memory usage information for given block(s) TYPE',
        metavar='TYPE',
        type=str,
        nargs='+',
    )
    groupDisplay.add_argument(
        '--showpids',
        help='Show current running KFD PIDs (pass details to VERBOSE for detailed information)',
        metavar='VERBOSE',
        const='summary',
        type=str,
        nargs='?',
    )
    groupDisplay.add_argument(
        '--showpidgpus',
        help='Show GPUs used by specified KFD PIDs (all if no arg given)',
        nargs='*',
    )
    groupDisplay.add_argument(
        '--showreplaycount',
        help='Show PCIe Replay Count',
        action='store_true',
    )
    groupDisplay.add_argument(
        '--showrasinfo',
        help='Show RAS enablement information and error counts for the specified block(s) (all if no arg given)',
        nargs='*',
    )
    groupDisplay.add_argument('--showvc', help='Show voltage curve', action='store_true')
    groupDisplay.add_argument(
        '--showxgmierr',
        help='Show XGMI error information since last read',
        action='store_true',
    )
    groupDisplay.add_argument(
        '--showtopo',
        help='Show hardware topology information',
        action='store_true',
    )
    groupDisplay.add_argument(
        '--showtopoaccess',
        help='Shows the link accessibility between GPUs ',
        action='store_true',
    )
    groupDisplay.add_argument(
        '--showtopoweight',
        help='Shows the relative weight between GPUs ',
        action='store_true',
    )
    groupDisplay.add_argument(
        '--showtopohops',
        help='Shows the number of hops between GPUs ',
        action='store_true',
    )
    groupDisplay.add_argument(
        '--showtopotype',
        help='Shows the link type between GPUs ',
        action='store_true',
    )
    groupDisplay.add_argument('--showtoponuma', help='Shows the numa nodes ', action='store_true')
    groupDisplay.add_argument(
        '--showenergycounter',
        help='Energy accumulator that stores amount of energy consumed',
        action='store_true',
    )
    groupDisplay.add_argument('--shownodesbw', help='Shows the numa nodes ', action='store_true')
    groupDisplay.add_argument(
        '--showcomputepartition',
        help='Shows current compute partitioning ',
        action='store_true',
    )
    groupDisplay.add_argument(
        '--showmemorypartition',
        help='Shows current memory partition ',
        action='store_true',
    )

    groupActionReset.add_argument(
        '-r',
        '--resetclocks',
        help='Reset clocks and OverDrive to default',
        action='store_true',
    )
    groupActionReset.add_argument(
        '--resetfans',
        help='Reset fans to automatic (driver) control',
        action='store_true',
    )
    groupActionReset.add_argument(
        '--resetprofile',
        help='Reset Power Profile back to default',
        action='store_true',
    )
    groupActionReset.add_argument(
        '--resetpoweroverdrive',
        help='Set the maximum GPU power back to the device default state',
        action='store_true',
    )
    groupActionReset.add_argument(
        '--resetxgmierr',
        help='Reset XGMI error count',
        action='store_true',
    )
    groupActionReset.add_argument(
        '--resetperfdeterminism',
        help='Disable performance determinism',
        action='store_true',
    )
    groupActionReset.add_argument(
        '--resetcomputepartition',
        help='Resets to boot compute partition state',
        action='store_true',
    )
    groupActionReset.add_argument(
        '--resetmemorypartition',
        help='Resets to boot memory partition state',
        action='store_true',
    )
    groupAction.add_argument(
        '--setclock',
        help='Set Clock Frequency Level(s) for specified clock (requires manual Perf level)',
        metavar=('TYPE', 'LEVEL'),
        nargs=2,
    )
    groupAction.add_argument(
        '--setsclk',
        help='Set GPU Clock Frequency Level(s) (requires manual Perf level)',
        type=int,
        metavar='LEVEL',
        nargs='+',
    )
    groupAction.add_argument(
        '--setmclk',
        help='Set GPU Memory Clock Frequency Level(s) (requires manual Perf level)',
        type=int,
        metavar='LEVEL',
        nargs='+',
    )
    groupAction.add_argument(
        '--setpcie',
        help='Set PCIE Clock Frequency Level(s) (requires manual Perf level)',
        type=int,
        metavar='LEVEL',
        nargs='+',
    )
    groupAction.add_argument(
        '--setslevel',
        help='Change GPU Clock frequency (MHz) and Voltage (mV) for a specific Level',
        metavar=('SCLKLEVEL', 'SCLK', 'SVOLT'),
        nargs=3,
    )
    groupAction.add_argument(
        '--setmlevel',
        help='Change GPU Memory clock frequency (MHz) and Voltage for (mV) a specific Level',
        metavar=('MCLKLEVEL', 'MCLK', 'MVOLT'),
        nargs=3,
    )
    groupAction.add_argument(
        '--setvc',
        help='Change SCLK Voltage Curve (MHz mV) for a specific point',
        metavar=('POINT', 'SCLK', 'SVOLT'),
        nargs=3,
    )
    groupAction.add_argument(
        '--setsrange',
        help='Set min and max SCLK speed',
        metavar=('SCLKMIN', 'SCLKMAX'),
        nargs=2,
    )
    groupAction.add_argument(
        '--setextremum',
        help='Set min/max of SCLK/MCLK speed',
        metavar=('min|max', 'sclk|mclk', 'CLK'),
        nargs=3,
    )
    groupAction.add_argument(
        '--setmrange',
        help='Set min and max MCLK speed',
        metavar=('MCLKMIN', 'MCLKMAX'),
        nargs=2,
    )
    groupAction.add_argument('--setfan', help='Set GPU Fan Speed (Level or %%)', metavar='LEVEL')
    groupAction.add_argument('--setperflevel', help='Set Performance Level', metavar='LEVEL')
    groupAction.add_argument(
        '--setoverdrive',
        help='Set GPU OverDrive level (requires manual|high Perf level)',
        metavar='%',
    )
    groupAction.add_argument(
        '--setmemoverdrive',
        help='Set GPU Memory Overclock OverDrive level (requires manual|high Perf level)',
        metavar='%',
    )
    groupAction.add_argument(
        '--setpoweroverdrive',
        help='Set the maximum GPU power using Power OverDrive in Watts',
        metavar='WATTS',
    )
    groupAction.add_argument(
        '--setprofile',
        help='Specify Power Profile level (#) or a quoted string of CUSTOM Profile attributes "# '
        '# # #..." (requires manual Perf level)',
    )
    groupAction.add_argument(
        '--setperfdeterminism',
        help='Set clock frequency limit to get minimal performance variation',
        type=int,
        metavar='SCLK',
        nargs=1,
    )
    groupAction.add_argument(
        '--setcomputepartition',
        help='Set compute partition',
        choices=compute_partition_type_l + [x.lower() for x in compute_partition_type_l],
        type=str,
        nargs=1,
    )
    groupAction.add_argument(
        '--setmemorypartition',
        help='Set memory partition',
        choices=memory_partition_type_l + [x.lower() for x in memory_partition_type_l],
        type=str,
        nargs=1,
    )
    groupAction.add_argument(
        '--rasenable',
        help='Enable RAS for specified block and error type',
        type=str,
        nargs=2,
        metavar=('BLOCK', 'ERRTYPE'),
    )
    groupAction.add_argument(
        '--rasdisable',
        help='Disable RAS for specified block and error type',
        type=str,
        nargs=2,
        metavar=('BLOCK', 'ERRTYPE'),
    )
    groupAction.add_argument(
        '--rasinject',
        help='Inject RAS poison for specified block (ONLY WORKS ON INSECURE BOARDS)',
        type=str,
        metavar='BLOCK',
        nargs=1,
    )
    groupActionGpuReset.add_argument(
        '--gpureset',
        help='Reset specified GPU (One GPU must be specified)',
        action='store_true',
    )

    groupFile.add_argument(
        '--load',
        help='Load Clock, Fan, Performance and Profile settings from FILE',
        metavar='FILE',
    )
    groupFile.add_argument(
        '--save',
        help='Save Clock, Fan, Performance and Profile settings to FILE',
        metavar='FILE',
    )

    groupResponse.add_argument(
        '--autorespond',
        help='Response to automatically provide for all prompts (NOT RECOMMENDED)',
        metavar='RESPONSE',
    )

    groupActionOutput.add_argument(
        '--loglevel',
        help='How much output will be printed for what program is doing, one of debug/info/warning/error/critical',
        metavar='LEVEL',
    )
    groupActionOutput.add_argument(
        '--json',
        help='Print output in JSON format',
        action='store_true',
    )
    groupActionOutput.add_argument('--csv', help='Print output in CSV format', action='store_true')

    args = parser.parse_args()

    # Must set PRINT_JSON early so the prints can be silenced
    if args.json or args.csv:
        PRINT_JSON = True

    # Initialize the rocm SMI library
    initializeRsmi()

    if args.version:
        showVersion(isCSV=args.csv)
        sys.exit()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)
    if args.loglevel is not None:
        numericLogLevel = getattr(logging, args.loglevel.upper(), logging.WARNING)
        logging.getLogger().setLevel(numericLogLevel)

    if (
        args.setsclk
        or args.setmclk
        or args.setpcie
        or args.resetfans
        or args.setfan
        or args.setperflevel
        or args.load
        or args.resetclocks
        or args.setprofile
        or args.resetprofile
        or args.setoverdrive
        or args.setmemoverdrive
        or args.setpoweroverdrive
        or args.resetpoweroverdrive
        or args.rasenable
        or args.rasdisable
        or args.rasinject
        or args.gpureset
        or args.setperfdeterminism
        or args.setslevel
        or args.setmlevel
        or args.setvc
        or args.setsrange
        or args.setextremum
        or args.setmrange
        or args.setclock
        or args.setcomputepartition
        or args.setmemorypartition
        or args.resetcomputepartition
        or args.resetmemorypartition
    ):
        relaunchAsSudo()

    # If there is one or more device specified, use that for all commands, otherwise use a
    # list of all available devices. Also use "is not None" as device 0 would
    # have args.device=0, and "if 0" returns false.
    if args.device is not None:
        deviceList = []
        for device in args.device:
            if not doesDeviceExist(device):
                logging.warning('No such device card%s', str(device))
                sys.exit()
            if device is None:
                printLog(None, 'ERROR: No DRM devices detected. Exiting', None)
                sys.exit()
            if (isAmdDevice(device) or args.alldevices) and device not in deviceList:
                deviceList.append(device)
    else:
        deviceList = listDevices()

    if deviceList is None:
        printLog(None, 'ERROR: No DRM devices available. Exiting', None)
        sys.exit(1)

    # If we want JSON/CSV output, initialize the keys (devices)
    if PRINT_JSON:
        for device in deviceList:
            JSON_DATA['card' + str(device)] = {}

    if not PRINT_JSON:
        print('\n')
    if not isConciseInfoRequested(args) and args.showhw == False:
        printLogSpacer(headerString)

    if args.showallinfo:
        args.list = True
        args.showid = True
        args.showvbios = True
        args.showdriverversion = True
        args.showfwinfo = 'all'
        args.showmclkrange = True
        args.showmemvendor = True
        args.showsclkrange = True
        args.showproductname = True
        args.showserial = True
        args.showuniqueid = True
        args.showvoltagerange = True
        args.showbus = True
        args.showpagesinfo = True
        args.showfan = True
        args.showpower = True
        args.showtemp = True
        args.showuse = True
        args.showenergycounter = True
        args.showmemuse = True
        args.showvoltage = True
        args.showclocks = True
        args.showmaxpower = True
        args.showmemoverdrive = True
        args.showoverdrive = True
        args.showperflevel = True
        args.showpids = 'summary'
        args.showpidgpus = []
        args.showreplaycount = True
        args.showvc = True
        args.showcomputepartition = True
        args.showmemorypartition = True

        if not PRINT_JSON:
            args.showprofile = True
            args.showclkfrq = True
            args.showclkvolt = True

    # Don't do reset in combination with any other command
    if args.gpureset:
        if not args.device:
            logging.error('No device specified. One device must be specified for GPU reset')
            printLogSpacer()
            sys.exit(1)
        logging.debug('Only executing GPU reset, no other commands will be executed')
        resetGpu(args.device)
        sys.exit(RETCODE)

    if not checkAmdGpus(deviceList):
        logging.warning('No AMD GPUs specified')

    if isConciseInfoRequested(args):
        showAllConcise(deviceList)
    if args.showhw:
        showAllConciseHw(deviceList)
    if args.showdriverversion:
        showDriverVersion(deviceList, rsmi_sw_component_t.RSMI_SW_COMP_DRIVER)
    if args.showtempgraph:
        showTempGraph(deviceList)
    if args.showid:
        showId(deviceList)
    if args.showuniqueid:
        showUId(deviceList)
    if args.showvbios:
        showVbiosVersion(deviceList)
    if args.showevents or str(args.showevents) == '[]':
        showEvents(deviceList, args.showevents)
    if args.resetclocks:
        resetClocks(deviceList)
    if args.showtemp:
        showCurrentTemps(deviceList)
    if args.showclocks:
        showCurrentClocks(deviceList)
    if args.showgpuclocks:
        showCurrentClocks(deviceList, 'sclk')
    if args.showfan:
        showCurrentFans(deviceList)
    if args.showperflevel:
        showPerformanceLevel(deviceList)
    if args.showoverdrive:
        showOverDrive(deviceList, 'sclk')
    if args.showmemoverdrive:
        showOverDrive(deviceList, 'mclk')
    if args.showmaxpower:
        showMaxPower(deviceList)
    if args.showprofile:
        showProfile(deviceList)
    if args.showpower:
        showPower(deviceList)
    if args.showclkfrq:
        showClocks(deviceList)
    if args.showuse:
        showGpuUse(deviceList)
    if args.showmemuse:
        showMemUse(deviceList)
    if args.showmemvendor:
        showMemVendor(deviceList)
    if args.showbw:
        showPcieBw(deviceList)
    if args.showreplaycount:
        showPcieReplayCount(deviceList)
    if args.showserial:
        showSerialNumber(deviceList)
    if args.showpids != None:
        showPids(args.showpids)
    if args.showpidgpus or str(args.showpidgpus) == '[]':
        showGpusByPid(args.showpidgpus)
    if args.showclkvolt:
        showPowerPlayTable(deviceList)
    if args.showvoltage:
        showVoltage(deviceList)
    if args.showbus:
        showBus(deviceList)
    if args.showmeminfo:
        showMemInfo(deviceList, args.showmeminfo)
    if args.showrasinfo or str(args.showrasinfo) == '[]':
        showRasInfo(deviceList, args.showrasinfo)
    # The second condition in the below 'if' statement checks whether showfwinfo was given arguments.
    # It compares itself to the string representation of the empty list and prints all firmwares.
    # This allows the user to call --showfwinfo without the 'all' argument and still print all.
    if args.showfwinfo or str(args.showfwinfo) == '[]':
        showFwInfo(deviceList, args.showfwinfo)
    if args.showproductname:
        showProduct(deviceList)
    if args.showxgmierr:
        showXgmiErr(deviceList)
    if args.shownodesbw:
        showNodesBw(deviceList)
    if args.showtopo:
        showHwTopology(deviceList)
    if args.showtopoaccess:
        showAccessibleTopology(deviceList)
    if args.showtopoweight:
        showWeightTopology(deviceList)
    if args.showtopohops:
        showHopsTopology(deviceList)
    if args.showtopotype:
        showTypeTopology(deviceList)
    if args.showtoponuma:
        showNumaTopology(deviceList)
    if args.showpagesinfo:
        showRetiredPages(deviceList)
    if args.showretiredpages:
        showRetiredPages(deviceList, 'reserved')
    if args.showpendingpages:
        showRetiredPages(deviceList, 'pending')
    if args.showunreservablepages:
        showRetiredPages(deviceList, 'unreservable')
    if args.showsclkrange:
        showRange(deviceList, 'sclk')
    if args.showmclkrange:
        showRange(deviceList, 'mclk')
    if args.showvoltagerange:
        showRange(deviceList, 'voltage')
    if args.showvc:
        showVoltageCurve(deviceList)
    if args.showenergycounter:
        showEnergy(deviceList)
    if args.showcomputepartition:
        showComputePartition(deviceList)
    if args.showmemorypartition:
        showMemoryPartition(deviceList)
    if args.setclock:
        setClocks(deviceList, args.setclock[0], [int(args.setclock[1])])
    if args.setsclk:
        setClocks(deviceList, 'sclk', args.setsclk)
    if args.setmclk:
        setClocks(deviceList, 'mclk', args.setmclk)
    if args.setpcie:
        setClocks(deviceList, 'pcie', args.setpcie)
    if args.setslevel:
        setPowerPlayTableLevel(
            deviceList,
            'sclk',
            args.setslevel[0],
            args.setslevel[1],
            args.setslevel[2],
            args.autorespond,
        )
    if args.setmlevel:
        setPowerPlayTableLevel(
            deviceList,
            'mclk',
            args.setmlevel[0],
            args.setmlevel[1],
            args.setmlevel[2],
            args.autorespond,
        )
    if args.resetfans:
        resetFans(deviceList)
    if args.setfan:
        setFanSpeed(deviceList, args.setfan)
    if args.setperflevel:
        setPerformanceLevel(deviceList, args.setperflevel)
    if args.setoverdrive:
        setClockOverDrive(deviceList, 'sclk', args.setoverdrive, args.autorespond)
    if args.setmemoverdrive:
        setClockOverDrive(deviceList, 'mclk', args.setmemoverdrive, args.autorespond)
    if args.setpoweroverdrive:
        setPowerOverDrive(deviceList, args.setpoweroverdrive, args.autorespond)
    if args.resetpoweroverdrive:
        resetPowerOverDrive(deviceList, args.autorespond)
    if args.setprofile:
        setProfile(deviceList, args.setprofile)
    if args.setvc:
        setVoltageCurve(deviceList, args.setvc[0], args.setvc[1], args.setvc[2], args.autorespond)
    if args.setextremum:
        setClockExtremum(
            deviceList,
            args.setextremum[0],
            args.setextremum[1],
            args.setextremum[2],
            args.autorespond,
        )
    if args.setsrange:
        setClockRange(deviceList, 'sclk', args.setsrange[0], args.setsrange[1], args.autorespond)
    if args.setmrange:
        setClockRange(deviceList, 'mclk', args.setmrange[0], args.setmrange[1], args.autorespond)
    if args.setperfdeterminism:
        setPerfDeterminism(deviceList, args.setperfdeterminism[0])
    if args.setcomputepartition:
        setComputePartition(deviceList, args.setcomputepartition[0])
    if args.setmemorypartition:
        setMemoryPartition(deviceList, args.setmemorypartition[0])
    if args.resetprofile:
        resetProfile(deviceList)
    if args.resetxgmierr:
        resetXgmiErr(deviceList)
    if args.resetperfdeterminism:
        resetPerfDeterminism(deviceList)
    if args.resetcomputepartition:
        resetComputePartition(deviceList)
    if args.resetmemorypartition:
        resetMemoryPartition(deviceList)
    if args.rasenable:
        setRas(deviceList, 'enable', args.rasenable[0], args.rasenable[1])
    if args.rasdisable:
        setRas(deviceList, 'disable', args.rasdisable[0], args.rasdisable[1])
    if args.rasinject:
        setRas(deviceList, 'inject', args.rasinject[0], args.rasinject[1])
    if args.load:
        load(args.load, args.autorespond)
    if args.save:
        save(deviceList, args.save)

    if RETCODE and not PRINT_JSON:
        logging.debug(' \t\t One or more commands failed.')
    # Set RETCODE value to 0, unless loglevel is None or 'warning' (default)
    if (
        args.loglevel is None
        or getattr(logging, args.loglevel.upper(), logging.WARNING) == logging.WARNING
    ):
        RETCODE = 0

    if PRINT_JSON:
        # Check that we have some actual data to print, instead of the
        # empty list that we initialized above
        for device in deviceList:
            if not JSON_DATA['card' + str(device)]:
                JSON_DATA.pop('card' + str(device))
        if not JSON_DATA:
            logging.warn('No JSON data to report')
            sys.exit(RETCODE)

        if not args.csv:
            print(json.dumps(JSON_DATA))
        else:
            devCsv = ''
            sysCsv = ''
            # JSON won't have any 'system' data without one of these flags
            if args.showdriverversion and args.showallinfo == False:
                sysCsv = formatCsv(['system'])
                print('%s' % (sysCsv))
            elif args.showallinfo is True:
                sysCsv = formatCsv(['system'])
                devCsv = formatCsv(deviceList)
                print(f'{sysCsv}\n{devCsv}')
            else:
                devCsv = formatCsv(deviceList)
                print(devCsv)

    if not isConciseInfoRequested(args) and args.showhw == False:
        printLogSpacer(footerString)

    rsmi_ret_ok(rocmsmi.rsmi_shut_down())
    exit(RETCODE)


def get_rocmsmi():
    return rocmsmi
