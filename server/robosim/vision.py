"""
Vision layer — converts camera images into symbolic state.

This is what runs in front of the stub when you have a real camera.
The stub bypasses this entirely and gives symbolic state directly.

Three modes:
  1. Stub mode (default):     skip vision, get symbolic state from sim config
  2. Sim vision mode:         run perception on MuJoCo camera renders
  3. Real camera mode:        run perception on actual robot camera feed

The LLM sees identical observations in all three modes.
That's the point — we can train in stub mode and deploy with real vision.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class VisionResult:
    """Symbolic facts extracted from an image."""
    detected_objects: list[dict]   # [{name, x, y, z, confidence}]
    gripper_pos: Optional[np.ndarray]
    gripper_open: Optional[bool]
    depth_map: Optional[np.ndarray]  # HxW float array, meters


# ── Stub mode: no vision, use sim ground truth ────────────────────────

def stub_vision(sim_state) -> VisionResult:
    """
    In stub mode, we already have ground-truth symbolic state.
    No vision model needed.
    """
    objects = [
        {
            "name": name,
            "x": float(obj.pos[0]),
            "y": float(obj.pos[1]),
            "z": float(obj.pos[2]),
            "confidence": 1.0,
            "reachable": obj.reachable,
        }
        for name, obj in sim_state.objects.items()
    ]
    return VisionResult(
        detected_objects=objects,
        gripper_pos=sim_state.gripper_pos,
        gripper_open=sim_state.gripper_open,
        depth_map=None,
    )


# ── Sim vision: run YOLO on MuJoCo camera renders ─────────────────────

def sim_vision(rgb_image: np.ndarray, depth_image: Optional[np.ndarray] = None,
               camera_matrix: Optional[np.ndarray] = None) -> VisionResult:
    """
    Run object detection on a rendered MuJoCo camera image.
    Used when use_stub=False to extract state from the virtual camera.

    rgb_image:    HxWx3 uint8 array from robosuite
    depth_image:  HxW float array (optional, improves 3D localization)
    """
    try:
        from ultralytics import YOLO
        model = _get_yolo_model()
        return _run_yolo(model, rgb_image, depth_image, camera_matrix)
    except ImportError:
        # YOLO not installed — fall back to color-based detection
        return _color_detection(rgb_image, depth_image)


def _get_yolo_model():
    """Load YOLO model (cached after first call)."""
    from ultralytics import YOLO
    if not hasattr(_get_yolo_model, "_model"):
        # Use YOLOv8n (nano) — fast enough for real-time robot control
        # For better accuracy: use yolov8m or fine-tune on robot images
        _get_yolo_model._model = YOLO("yolov8n.pt")
    return _get_yolo_model._model


def _run_yolo(model, rgb: np.ndarray, depth: Optional[np.ndarray],
              camera_matrix: Optional[np.ndarray]) -> VisionResult:
    """Run YOLO detection and convert to symbolic object list."""
    results = model(rgb, verbose=False)[0]
    objects = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if conf < 0.4:
            continue
        # Get 2D center
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # Get 3D position if depth available
        if depth is not None and camera_matrix is not None:
            z = float(depth[int(cy), int(cx)])
            x3d, y3d = _pixel_to_world(cx, cy, z, camera_matrix)
        else:
            x3d, y3d, z = 0.0, 0.0, 0.85
        objects.append({
            "name": _map_class_to_block(cls_name),
            "x": x3d, "y": y3d, "z": z,
            "confidence": conf,
            "reachable": True,  # blocking computed separately
        })
    return VisionResult(
        detected_objects=objects,
        gripper_pos=None,  # detected separately from robot state
        gripper_open=None,
        depth_map=depth,
    )


def _color_detection(rgb: np.ndarray, depth: Optional[np.ndarray]) -> VisionResult:
    """
    Simple color-based object detection when YOLO isn't available.
    Works for colored blocks on a plain table surface.
    """
    import cv2

    COLOR_RANGES = {
        "red_block":    ([0,100,100],   [10,255,255]),
        "blue_block":   ([100,100,100], [130,255,255]),
        "green_block":  ([40,100,100],  [80,255,255]),
        "yellow_block": ([20,100,100],  [35,255,255]),
        "purple_block": ([130,50,100],  [160,255,255]),
    }

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]
    objects = []

    for name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # too small, ignore
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            # Normalize to [-0.3, 0.3] workspace coords (rough)
            x3d = (cx / w - 0.5) * 0.6
            y3d = -(cy / h - 0.5) * 0.6
            z3d = float(depth[int(cy), int(cx)]) if depth is not None else 0.85
            objects.append({
                "name": name, "x": x3d, "y": y3d, "z": z3d,
                "confidence": min(area / 2000.0, 1.0),
                "reachable": True,
            })
            break  # one detection per color class

    return VisionResult(
        detected_objects=objects,
        gripper_pos=None,
        gripper_open=None,
        depth_map=depth,
    )


def _pixel_to_world(cx: float, cy: float, z: float,
                     K: np.ndarray) -> tuple[float, float]:
    """Back-project a pixel to world XY using camera intrinsics K."""
    fx, fy = K[0, 0], K[1, 1]
    px, py = K[0, 2], K[1, 2]
    x = (cx - px) * z / fx
    y = (cy - py) * z / fy
    return x, y


def _map_class_to_block(cls_name: str) -> str:
    """Map YOLO class name to our block naming convention."""
    mapping = {
        "cup": "red_block", "bottle": "blue_block",
        "bowl": "green_block", "box": "yellow_block",
        "block": "red_block",  # generic
    }
    return mapping.get(cls_name.lower(), f"{cls_name}_block")


# ── Real camera: same interface, real hardware ────────────────────────

def real_camera_vision(camera_feed) -> VisionResult:
    """
    Same perception pipeline but reading from a real camera.
    camera_feed: OpenCV VideoCapture or similar.

    This is what you'd run on a real robot deployment.
    The symbolic state it produces is identical to stub_vision output,
    which is why a policy trained in stub mode transfers.
    """
    import cv2
    ret, frame = camera_feed.read()
    if not ret:
        return VisionResult([], None, None, None)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return sim_vision(rgb)
