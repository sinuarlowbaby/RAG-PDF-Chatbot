import platform
import subprocess
import socket
import psutil
import json

def get_integrated_and_shared_vram():
    """Extracts integrated VRAM, shared system allocations, or Unified Memory configurations."""
    system_os = platform.system()
    gpu_metrics = []

    # --- 1. Apple Silicon (Unified Memory Environment) ---
    if system_os == "Darwin":
        try:
            # Check if Apple Silicon (M1/M2/M3/M4/M5 etc.)
            cpu_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if "Apple" in cpu_brand:
                # Read hardware unified memory limits allocated for graphics
                total_ram = psutil.virtual_memory().total / (1024 ** 3)
                # macOS usually allocates up to 75% of unified memory to the GPU by default
                default_gpu_max = total_ram * 0.75 
                gpu_metrics.append({
                    "Type": "Unified Memory Architecture (Apple Silicon)",
                    "Hardware System": cpu_brand,
                    "Total Unified RAM (GB)": round(total_ram, 2),
                    "Estimated Max Usable VRAM (GB)": round(default_gpu_max, 2),
                    "Context": "Shared seamlessly between CPU and GPU without PCIe overhead."
                })
                return gpu_metrics
        except Exception:
            pass

    # --- 2. Windows (Integrated & Shared RAM allocations via WMI) ---
    if system_os == "Windows":
        try:
            # Queries the Win32_VideoController for hardware layer specs
            cmd = "wmic path win32_VideoController get Name, AdapterRAM /format:csv"
            output = subprocess.check_output(cmd, shell=True, text=True).strip()
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            
            for line in lines[1:]: # Skip CSV schema header
                parts = line.split(',')
                if len(parts) >= 3 and parts[2].isdigit():
                    name = parts[1]
                    raw_ram = int(parts[2])
                    vram_gb = round(raw_ram / (1024 ** 3), 2)
                    
                    # Distinguish common integrated chips
                    is_integrated = any(x in name.lower() for x in ["intel", "amd radeon(tm) graphics", "iris", "graphics"])
                    
                    # Note: Windows often reports a placeholder like 128MB/512MB for iGPUs,
                    # while the rest is drawn dynamically from system memory pools.
                    total_sys_ram = psutil.virtual_memory().total / (1024 ** 3)
                    max_shared_pool = total_sys_ram * 0.50 # Windows typical cap for shared memory
                    
                    gpu_metrics.append({
                        "GPU Card Name": name,
                        "Type": "Integrated" if is_integrated else "Discrete",
                        "Reported Hardware VRAM (GB)": vram_gb if vram_gb > 0 else "Dynamic (<0.5GB)",
                        "Max Dynamic Shared Allocation (GB)": round(max_shared_pool, 2) if is_integrated else "N/A"
                    })
            if gpu_metrics:
                return gpu_metrics
        except Exception:
            pass

    # --- 3. Linux Fallback (sysfs / glxinfo) ---
    if system_os == "Linux":
        try:
            # Read graphic details using standard Mesa drivers
            output = subprocess.check_output("glxinfo -B", shell=True, text=True)
            for line in output.split('\n'):
                if "Video memory" in line or "Unified memory" in line:
                    gpu_metrics.append({
                        "Type": "Linux Graphics Subsystem",
                        "Memory String": line.strip()
                    })
            if gpu_metrics:
                return gpu_metrics
        except Exception:
            gpu_metrics.append({"Note": "Integrated card detected. Check shared system RAM directly."})
            
    return gpu_metrics if gpu_metrics else [{"Status": "No separate integrated metrics found"}]

def get_complete_specs():
    svmem = psutil.virtual_memory()
    return {
        "OS": platform.system(),
        "Architecture": platform.machine(),
        "Available System RAM (GB)": round(svmem.available / (1024 ** 3), 2),
        "Graphics Memory Context": get_integrated_and_shared_vram()
    }

if __name__ == "__main__":
    print(json.dumps(get_complete_specs(), indent=4))
