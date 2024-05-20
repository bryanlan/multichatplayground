import subprocess
import time
import threading

class GPUTimeKeeper:
    def __init__(self):
        self.gpu_type = self._detect_gpu_type()
        self.worker_thread = None
        self.running = False
   
        self.execution_time_ms = 0

    def _detect_gpu_type(self):
        try:
            # Check for NVIDIA GPU
            subprocess.check_output(['nvidia-smi'])
            return 'nvidia'
        except FileNotFoundError:
            # Check for AMD GPU (stub)
            # You can add the actual detection logic for AMD GPUs here
            return 'amd'
        except Exception:
            # Check for Intel GPU (stub)
            # You can add the actual detection logic for Intel GPUs here
            return 'intel'

    def _get_nvidia_gpu_utilization(self):
        # Run the nvidia-smi command and capture the output
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
        output = output.decode('utf-8')
        utilization_gpu = int(output.strip())
        return utilization_gpu

    def _get_amd_gpu_utilization(self):
        # Stub for AMD GPU utilization
        # You can add the actual logic to retrieve AMD GPU utilization here
        return 0

    def _get_intel_gpu_utilization(self):
        # Stub for Intel GPU utilization
        # You can add the actual logic to retrieve Intel GPU utilization here
        return 0

    def _monitor_gpu_utilization(self):
        start_time = time.time()
        while self.running:
            if self.gpu_type == 'nvidia':
                utilization_gpu = self._get_nvidia_gpu_utilization()
            elif self.gpu_type == 'amd':
                utilization_gpu = self._get_amd_gpu_utilization()
            elif self.gpu_type == 'intel':
                utilization_gpu = self._get_intel_gpu_utilization()

            if utilization_gpu > self.threshold:
                current_time = time.time()
                self.execution_time_ms += (current_time - start_time) * 1000
                start_time = current_time

            time.sleep(0.1)

    def start_timer(self, threshold):
        if not self.running:
            self.threshold = threshold
            self.execution_time_ms = 0
            self.running = True
            self.worker_thread = threading.Thread(target=self._monitor_gpu_utilization)
            self.worker_thread.start()
        else:
            print("Timer is already running. Stop the timer before starting a new one.")

    def stop_timer(self):
        if self.running:
            self.running = False
            self.worker_thread.join()
            execution_time_ms = self.execution_time_ms
            self.execution_time_ms = 0
            return execution_time_ms
        else:
            print("Timer is not running. Start the timer before stopping it.")
            return 0