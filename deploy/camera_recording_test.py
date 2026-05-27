# ruff: noqa: E402, N802, SIM115
import multiprocessing as mp
import sys
import types

import numpy as np


class _FakeVideoWriter:
    def __init__(self, path, _fourcc, _fps, _size):
        self._file = open(path, "wb")

    def isOpened(self):
        return True

    def write(self, frame):
        self._file.write(np.asarray(frame).tobytes())

    def release(self):
        self._file.close()


sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        CAP_V4L2=0,
        CAP_PROP_BUFFERSIZE=0,
        CAP_PROP_FPS=0,
        CAP_PROP_FRAME_HEIGHT=0,
        CAP_PROP_FRAME_WIDTH=0,
        CAP_PROP_POS_MSEC=0,
        VideoCapture=object,
        VideoWriter=_FakeVideoWriter,
        VideoWriter_fourcc=lambda *args: 0,
        setNumThreads=lambda _num_threads: None,
    ),
)
sys.modules.setdefault(
    "deploy.umi.shared_memory.shared_memory_ring_buffer",
    types.SimpleNamespace(SharedMemoryRingBuffer=object),
)
sys.modules.setdefault(
    "threadpoolctl",
    types.SimpleNamespace(threadpool_limits=lambda _num_threads: None),
)

from deploy.umi.real_world.camera.multi_uvc_camera import MultiUvcCamera
from deploy.umi.real_world.camera.uvc_camera import Command
from deploy.umi.real_world.camera.uvc_camera import FrameVideoRecorder
from deploy.umi.real_world.camera.uvc_camera import UvcCamera


def test_frame_video_recorder_writes_mp4_and_timestamp_sidecar(tmp_path):
    video_path = tmp_path / "camera0.mp4"
    recorder = FrameVideoRecorder(video_path=video_path, fps=10)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[..., 1] = 128

    recorder.write_frame(
        frame,
        frame_metadata={
            "frame_idx": 3,
            "timestamp": 100.25,
            "camera_capture_timestamp": 100.2,
            "camera_receive_timestamp": 100.3,
        },
    )
    recorder.close()

    assert video_path.exists()
    assert video_path.stat().st_size > 0
    sidecar_path = tmp_path / "camera0.timestamps.jsonl"
    assert sidecar_path.exists()
    assert sidecar_path.read_text().splitlines() == [
        ('{"camera_capture_timestamp": 100.2, "camera_receive_timestamp": 100.3, "frame_idx": 3, "timestamp": 100.25}')
    ]


def test_uvc_camera_start_recording_sends_process_command(tmp_path):
    camera = object.__new__(UvcCamera)
    camera.command_queue = mp.Queue()
    camera.recording_done_event = mp.Event()
    camera.recording_done_event.set()

    camera.start_recording(video_path=tmp_path / "camera0.mp4", start_time=123.0)
    command = camera.command_queue.get(timeout=1.0)

    assert command["cmd"] == Command.START_RECORDING.value
    assert command["video_path"] == str(tmp_path / "camera0.mp4")
    assert command["start_time"] == 123.0
    assert not camera.recording_done_event.is_set()


class _FakeCamera:
    def __init__(self):
        self.started = None
        self.stopped = False

    def start_recording(self, *, video_path, start_time):
        self.started = {"video_path": video_path, "start_time": start_time}

    def stop_recording(self):
        self.stopped = True


def test_multi_uvc_camera_dispatches_recording_paths_per_camera(tmp_path):
    camera = object.__new__(MultiUvcCamera)
    camera.cameras = [_FakeCamera(), _FakeCamera()]
    video_paths = [tmp_path / "0.mp4", tmp_path / "1.mp4"]

    camera.start_recording(video_path=video_paths, start_time=456.0)
    camera.stop_recording()

    assert camera.cameras[0].started == {"video_path": video_paths[0], "start_time": 456.0}
    assert camera.cameras[1].started == {"video_path": video_paths[1], "start_time": 456.0}
    assert camera.cameras[0].stopped
    assert camera.cameras[1].stopped
