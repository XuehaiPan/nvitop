# nvitop Python script to log stats for all Nvidia GPUs, physical and MIG.
# You must run this within a venv that has nvitop installed.
# by Dan MacDonald 2026

import csv
import sys
import platform
from datetime import datetime
from nvitop import PhysicalDevice, MigDevice, NA
import pynvml

def collect_gpu_stats():
    writer = csv.writer(sys.stdout)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kernel_version = platform.release()

    try:
        # Initialize pynvml to get the driver version string correctly
        pynvml.nvmlInit()
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode('utf-8')

        phys_devices = PhysicalDevice.all()
        for p_dev in phys_devices:
            m_devices = p_dev.mig_devices()
            if m_devices:
                for m_dev in m_devices:
                    write_row(writer, timestamp, kernel_version, driver_version, m_dev, parent=p_dev)
            else:
                write_row(writer, timestamp, kernel_version, driver_version, p_dev, parent=p_dev)
    except Exception as e:
        print(f"Init error: {e}", file=sys.stderr)
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def write_row(writer, timestamp, kernel, driver, device, parent):
    try:
        pci = parent.pci_info()
        pci_address = pci.busId.decode('utf-8') if isinstance(pci.busId, bytes) else pci.busId
        temp = parent.temperature()

        # --- PROCESS-BASED GPU UTILIZATION ---
        gpu_util = 0
        try:
            # First, check if there are any active processes on this specific MIG slice
            procs = device.processes()
            if procs and len(procs) > 0:
                # If processes exist, we know the slice is being utilized.
                gpu_util = 1
            else:
                # Fallback to standard query if no processes but hardware says something
                val = device.gpu_utilization()
                gpu_util = val if (val is not None and str(val) != 'N/A') else 0
        except:
            gpu_util = 0

        mem_used = device.memory_used() / (1024**2)
        mem_total = device.memory_total() / (1024**2)
        uuid = device.uuid()

        writer.writerow([
            timestamp,
            kernel,
            driver,
            device.index,
            pci_address,
            temp,
            gpu_util,
            f"{mem_used:.2f}",
            f"{mem_total:.2f}",
            uuid
        ])
    except Exception as e:
        idx = getattr(device, 'index', 'unknown')
        print(f"Error on device {idx}: {e}", file=sys.stderr)

if __name__ == "__main__":
    collect_gpu_stats()
