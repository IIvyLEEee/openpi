from typing import Callable, Dict, List, Optional

import numpy as np
from multiprocessing.managers import SharedMemoryManager

from deploy.umi.real_world.camera.uvc_camera import UvcCamera


class MultiUvcCamera:
    def __init__(
        self,
        dev_video_paths: List[str],
        shm_manager: Optional[SharedMemoryManager] = None,
        resolution=(1920, 1080),
        capture_fps=60,
        put_fps=None,
        put_downsample=False,
        get_max_k=30,
        receive_latency=0.0,
        cap_buffer_size=1,
        transform: Optional[List[Callable[[Dict], Dict]]] = None,
        output_resolution=None,
        verbose=False,
    ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        n_cameras = len(dev_video_paths)
        resolution = _repeat_to_list(resolution, n_cameras)
        capture_fps = _repeat_to_list(capture_fps, n_cameras)
        cap_buffer_size = _repeat_to_list(cap_buffer_size, n_cameras)
        transform = _repeat_to_list(transform, n_cameras)
        output_resolution = _repeat_to_list(output_resolution, n_cameras)

        self.cameras = [
            UvcCamera(
                shm_manager=shm_manager,
                dev_video_path=path,
                resolution=resolution[i],
                capture_fps=capture_fps[i],
                put_fps=put_fps,
                put_downsample=put_downsample,
                get_max_k=get_max_k,
                receive_latency=receive_latency,
                cap_buffer_size=cap_buffer_size[i],
                transform=transform[i],
                output_resolution=output_resolution[i],
                verbose=verbose,
            )
            for i, path in enumerate(dev_video_paths)
        ]

    @property
    def n_cameras(self):
        return len(self.cameras)

    @property
    def is_ready(self):
        return all(camera.is_ready for camera in self.cameras)

    def start(self, wait=True, put_start_time=None):
        for camera in self.cameras:
            camera.start(wait=False, put_start_time=put_start_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for camera in self.cameras:
            camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras:
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras:
            camera.join(timeout=5.0)

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        if out is None:
            out = {}
        for i, camera in enumerate(self.cameras):
            this_out = out.get(i)
            out[i] = camera.get(k=k, out=this_out)
        return out

    def get_vis(self, k=None, out=None):
        return self.get(k=k, out=out)

    def start_recording(self, *, video_path, start_time=None):
        video_paths = _repeat_to_list(video_path, self.n_cameras)
        for camera, path in zip(self.cameras, video_paths):
            camera.start_recording(video_path=path, start_time=start_time)

    def stop_recording(self):
        for camera in self.cameras:
            camera.stop_recording()

    def restart_put(self, start_time):
        for camera in self.cameras:
            camera.restart_put(start_time)


def _repeat_to_list(value, n):
    if isinstance(value, list):
        assert len(value) == n
        return value
    return [value for _ in range(n)]
