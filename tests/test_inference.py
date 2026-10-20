import pytest
import numpy as np
from PIL import Image
import torch
from morphic_cv import MorphicInferenceEngine


def test_engine_initialization():
    engine = MorphicInferenceEngine(
        model_name="dinov2_vits14",
        class_names=["cat", "dog"],
        device="cpu"
    )
    assert engine.num_classes == 2
    assert engine.class_names == ["cat", "dog"]
    assert engine.device.type == "cpu"


def test_engine_single_prediction():
    engine = MorphicInferenceEngine(
        model_name="dinov2_vits14",
        class_names=["cat", "dog"],
        device="cpu"
    )
    # Create synthetic test image
    img = Image.new("RGB", (256, 256), color=(128, 64, 32))
    res = engine.predict(img, return_probs=True)

    assert "label" in res
    assert res["label"] in ["cat", "dog"]
    assert 0.0 <= res["confidence"] <= 100.0
    assert res["latency_ms"] > 0.0
    assert "probabilities" in res
    assert len(res["probabilities"]) == 2


def test_engine_batch_prediction():
    engine = MorphicInferenceEngine(
        model_name="dinov2_vits14",
        class_names=["A", "B", "C"],
        device="cpu"
    )
    images = [
        Image.new("RGB", (224, 224), color=(i * 40, i * 20, 100))
        for i in range(4)
    ]
    results = engine.predict_batch(images, batch_size=2)
    assert len(results) == 4
    for r in results:
        assert r["label"] in ["A", "B", "C"]
        assert 0.0 <= r["confidence"] <= 100.0


def test_engine_salient_objects():
    engine = MorphicInferenceEngine(
        model_name="dinov2_vits14",
        class_names=["object"],
        device="cpu"
    )
    # Image with high contrast shapes
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[50:150, 50:150] = 255  # bright square
    img = Image.fromarray(arr)

    detections = engine.detect_salient_objects(img, threshold=0.1, max_boxes=3)
    assert isinstance(detections, list)
    if detections:
        box = detections[0]["bbox"]
        assert len(box) == 4
        assert detections[0]["label"] == "object"


def test_engine_face_swap():
    engine = MorphicInferenceEngine(
        model_name="dinov2_vits14",
        class_names=["face"],
        device="cpu"
    )
    # Synthetic user frame and celebrity frame
    user_frame = np.full((320, 320, 3), 160, dtype=np.uint8)
    celeb_frame = np.full((180, 180, 3), 210, dtype=np.uint8)

    # Test enrolling celebrity
    success = engine.enroll_celebrity("brad_pitt", celeb_frame)
    assert success is True
    assert "brad_pitt" in engine.celebrity_bank

    # Test swap_face
    swapped = engine.swap_face(user_frame, "brad_pitt")
    assert isinstance(swapped, np.ndarray)
    assert swapped.shape == (320, 320, 3)

