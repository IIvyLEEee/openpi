import enum
import multiprocessing as mp
import time
from typing import Callable, Dict, Optional

import cv2
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from threadpoolctl import threadpool_limits

from deploy.umi.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from deploy.umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Command(enum.Enum):
    RESTART_PUT = 0


class UvcCamera(mp.Process):
    """Minimal UVC camera process for GoPro/capture-card deployment."""

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        dev_video_path: str,
        resolution=(1920, 1080),
        capture_fps=60,
        put_fps=None,
        put_downsample=False,
        get_max_k=30,
        receive_latency=0.0,
        cap_buffer_size=1,
        transform: Optional[Callable[[Dict], Dict]] = None,
        output_resolution=None,
        num_threads=2,
        verbose=False,
    ):
        super().__init__()
        if put_fps is None:
            put_fps = capture_fps

        color_resolution = tuple(output_resolution) if output_resolution is not None else tuple(resolution)
        examples = {
            "color": np.empty(color_resolution[::-1] + (3,), dtype=np.uint8),
            "camera_capture_timestamp": 0.0,
            "camera_receive_timestamp": 0.0,
            "timestamp": 0.0,
            "step_idx": 0,
        }

        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        self.dev_video_path = dev_video_path
        self.resolution = tuple(resolution)
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.cap_buffer_size = cap_buffer_size
        self.transform = transform
        self.num_threads = num_threads
        self.verbose = verbose
        self.put_start_time = None

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.daemon = True

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def start_wait(self, timeout=10.0):
        if not self.ready_event.wait(timeout=timeout):
            raise RuntimeError(f"Timed out waiting for UVC camera {self.dev_video_path} to start.")

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join(timeout=5.0)

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k, out=out)

    def restart_put(self, start_time):
        self.put_start_time = start_time

    def start_recording(self, *args, **kwargs):
        pass

    def stop_recording(self):
        pass

    def run(self):
        threadpool_limits(self.num_threads)
        cv2.setNumThreads(self.num_threads)

        cap = cv2.VideoCapture(self.dev_video_path, cv2.CAP_V4L2)
        try:
            width, height = self.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cap_buffer_size)
            cap.set(cv2.CAP_PROP_FPS, self.capture_fps)

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open UVC camera {self.dev_video_path}")

            put_idx = None
            put_start_time = self.put_start_time or time.time()
            iter_idx = 0
            t_start = time.time()

            while not self.stop_event.is_set():
                ret = cap.grab()
                if not ret:
                    raise RuntimeError(f"Failed to grab frame from {self.dev_video_path}")

                ret, frame = cap.retrieve()
                t_recv = time.time()
                if not ret:
                    raise RuntimeError(f"Failed to retrieve frame from {self.dev_video_path}")

                t_cap = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                t_cap = t_cap - time.monotonic() + time.time()
                t_cal = t_recv - self.receive_latency

                data = {
                    "camera_receive_timestamp": t_recv,
                    "camera_capture_timestamp": t_cap,
                    "timestamp": t_cal,
                    "color": frame,
                }
                put_data = self.transform(dict(data)) if self.transform is not None else data

                if self.put_downsample:
                    _, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[t_cal],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        next_global_idx=put_idx,
                        allow_negative=True,
                    )
                    for step_idx in global_idxs:
                        put_data["step_idx"] = step_idx
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    put_data["step_idx"] = int((t_cal - put_start_time) * self.put_fps)
                    self.ring_buffer.put(put_data, wait=False)

                if iter_idx == 0:
                    self.ready_event.set()

                if self.verbose:
                    now = time.time()
                    print(f"[UvcCamera {self.dev_video_path}] FPS {1 / max(now - t_start, 1e-6):.1f}")
                    t_start = now

                iter_idx += 1
        except Exception:
            import traceback

            traceback.print_exc()
        finally:
            cap.release()
