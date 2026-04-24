import torch
from unittest.mock import patch

from comfy.clip_vision import ClipVisionModel
import comfy.clip_vision as clip_vision


class DummyVisionModel:
    def __call__(self, pixel_values, intermediate_output):
        batch_size = pixel_values.shape[0]
        return (
            torch.zeros((batch_size, 1, 1, 1), dtype=torch.float32),
            torch.zeros((batch_size, 1, 1), dtype=torch.float32),
            torch.zeros((batch_size, 1), dtype=torch.float32),
            torch.zeros((batch_size, 1), dtype=torch.float32),
        )


def make_clip_vision_model(model_type: str, image_size: int = 224) -> ClipVisionModel:
    model = ClipVisionModel.__new__(ClipVisionModel)
    model.load_device = torch.device("cpu")
    model.image_mean = [0.5, 0.5, 0.5]
    model.image_std = [0.5, 0.5, 0.5]
    model.image_size = image_size
    model.model_type = model_type
    model.config = {"patch_size": 16, "num_patches": 256}
    model.return_all_hidden_states = False
    model.model = DummyVisionModel()
    model.patcher = object()
    return model


class TestClipVisionSourceRestorePreprocess:
    def test_non_birefnet_ignores_source_image_samples(self):
        model = make_clip_vision_model("siglip2_vision_model", image_size=-1)
        image = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
        image.source_image_samples = [
            torch.zeros((1, 5, 7, 3), dtype=torch.float32),
            torch.zeros((1, 6, 4, 3), dtype=torch.float32),
        ]

        def fake_siglip2_preprocess(sample, **kwargs):
            assert sample.shape == (2, 8, 8, 3)
            return torch.zeros((sample.shape[0], 3, 16, 16), dtype=torch.float32)

        with patch.object(clip_vision.comfy.model_management, "load_model_gpu"), \
             patch.object(clip_vision.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")), \
             patch.object(clip_vision.comfy.clip_model, "siglip2_preprocess", side_effect=fake_siglip2_preprocess) as preprocess:
            outputs = model.encode_image(image, crop=False)

        assert preprocess.call_count == 1
        assert outputs["source_image_sizes"] == [(8, 8), (8, 8)]

    def test_birefnet_uses_source_image_samples_for_restore_metadata(self):
        model = make_clip_vision_model("birefnet")
        image = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
        image.source_image_samples = [
            torch.zeros((1, 5, 7, 3), dtype=torch.float32),
            torch.zeros((1, 6, 4, 3), dtype=torch.float32),
        ]

        def fake_clip_preprocess(sample, **kwargs):
            assert sample.shape[0] == 1
            return torch.zeros((sample.shape[0], 3, 4, 4), dtype=torch.float32)

        with patch.object(clip_vision.comfy.model_management, "load_model_gpu"), \
             patch.object(clip_vision.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")), \
             patch.object(clip_vision.comfy.clip_model, "clip_preprocess", side_effect=fake_clip_preprocess) as preprocess:
            outputs = model.encode_image(image, crop=False)

        assert preprocess.call_count == 2
        assert outputs["source_image_sizes"] == [(5, 7), (6, 4)]
