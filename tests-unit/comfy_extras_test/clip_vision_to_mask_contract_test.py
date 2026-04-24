import importlib
import sys
import pytest
import torch
from unittest.mock import MagicMock, patch

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()


def _import_test_modules():
    existing_module_names = set(sys.modules)
    with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
        imported_modules = (
            importlib.import_module("comfy_extras.nodes_mask"),
            importlib.import_module("comfy_extras.nodes_images"),
            importlib.import_module("comfy_extras.nodes_post_processing"),
        )
        new_modules = {
            name: module
            for name, module in sys.modules.items()
            if name not in existing_module_names and name not in {"nodes", "server"}
        }
    sys.modules.update(new_modules)
    return imported_modules


nodes_mask, nodes_images, nodes_post_processing = _import_test_modules()

ClipVisionToMask = nodes_mask.ClipVisionToMask
MaskToImage = nodes_mask.MaskToImage
ImageFromBatch = nodes_images.ImageFromBatch
batch_images = nodes_post_processing.batch_images


class FakeClipVisionOutput:
    def __init__(self, **values):
        self.__dict__.update(values)

    def __getitem__(self, key):
        return getattr(self, key)


class TestClipVisionToMaskContract:
    def test_non_birefnet_layout_raises(self):
        clip_vision_output = FakeClipVisionOutput(
            last_hidden_state=torch.zeros((1, 257, 1024), dtype=torch.float32),
            source_image_sizes=[(64, 64)],
            clip_vision_model_type="clip_vision_model",
        )

        with pytest.raises(ValueError, match="ClipVisionToMask source restore requires a BiRefNet clip vision output"):
            ClipVisionToMask.execute(clip_vision_output)

    def test_source_size_length_mismatch_raises(self):
        clip_vision_output = FakeClipVisionOutput(
            last_hidden_state=torch.zeros((2, 1, 4, 4), dtype=torch.float32),
            source_image_sizes=[(8, 8)],
            clip_vision_model_type="birefnet",
        )

        with pytest.raises(ValueError, match="ClipVisionToMask source_image_sizes length must equal batch size"):
            ClipVisionToMask.execute(clip_vision_output)

    def test_uncropped_source_restore(self):
        clip_vision_output = FakeClipVisionOutput(
            last_hidden_state=torch.zeros((2, 1, 4, 4), dtype=torch.float32),
            source_image_sizes=[(6, 6), (6, 6)],
            clip_vision_model_type="birefnet",
            source_restore_crop_mode="none",
            preprocess_image_sizes=[(4, 4), (4, 4)],
        )

        def fake_upscale(sample, width, height, method, crop):
            assert sample.shape == (2, 1, 4, 4)
            assert (width, height, method, crop) == (6, 6, "bilinear", "disabled")
            return torch.full((sample.shape[0], sample.shape[1], height, width), 0.75, dtype=sample.dtype)

        with patch.object(nodes_mask.comfy.utils, "common_upscale", side_effect=fake_upscale) as common_upscale:
            result = ClipVisionToMask.execute(clip_vision_output)

        assert result[0].shape == (2, 1, 6, 6)
        assert common_upscale.call_count == 1

    def test_mixed_batch_source_restore_is_deferred_until_image_from_batch(self):
        clip_vision_output = FakeClipVisionOutput(
            last_hidden_state=torch.zeros((2, 1, 4, 4), dtype=torch.float32),
            source_image_sizes=[(6, 6), (2, 3)],
            clip_vision_model_type="birefnet",
            source_restore_crop_mode="none",
            preprocess_image_sizes=[(4, 4), (4, 4)],
        )

        mask = ClipVisionToMask.execute(clip_vision_output)[0]
        assert mask.shape == (2, 1, 4, 4)
        assert mask.source_image_sizes == [(6, 6), (2, 3)]

        image = MaskToImage.execute(mask)[0]

        def fake_upscale(sample, width, height, method, crop):
            assert (width, height, method, crop) == (3, 2, "bilinear", "disabled")
            return torch.full((sample.shape[0], sample.shape[1], height, width), 0.5, dtype=sample.dtype)

        with patch.object(nodes_images.comfy.utils, "common_upscale", side_effect=fake_upscale) as common_upscale:
            result = ImageFromBatch.execute(image, 1, 1)

        assert result[0].shape == (1, 2, 3, 3)
        assert common_upscale.call_count == 1

    def test_center_crop_restore_is_skipped_after_image_from_batch(self):
        clip_vision_output = FakeClipVisionOutput(
            last_hidden_state=torch.zeros((1, 1, 4, 4), dtype=torch.float32),
            source_image_sizes=[(8, 6)],
            clip_vision_model_type="birefnet",
            source_restore_crop_mode="center",
            preprocess_image_sizes=[(4, 4)],
        )

        mask = ClipVisionToMask.execute(clip_vision_output)[0]
        image = MaskToImage.execute(mask)[0]

        with patch.object(nodes_images.comfy.utils, "common_upscale") as common_upscale:
            result = ImageFromBatch.execute(image, 0, 1)

        assert result[0].shape == (1, 4, 4, 3)
        assert common_upscale.call_count == 0


class TestBatchImagesSourceRestoreMetadata:
    def test_uniform_batch_omits_source_image_samples(self):
        first = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        second = torch.ones((1, 4, 4, 3), dtype=torch.float32)

        batched = batch_images([first, second])

        assert batched is not None
        assert batched.source_image_sizes == [(4, 4), (4, 4)]
        assert getattr(batched, "source_image_samples", None) is None
        assert getattr(batched, "preprocess_image_sizes", None) is None

    def test_mixed_batch_preserves_source_image_samples(self):
        first = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        second = torch.ones((1, 6, 5, 3), dtype=torch.float32)

        batched = batch_images([first, second])

        assert batched is not None
        assert batched.source_image_sizes == [(4, 4), (6, 5)]
        assert batched.shape == (2, 4, 4, 3)
        assert len(batched.source_image_samples) == 2
        assert batched.source_image_samples[0].shape == (1, 4, 4, 3)
        assert batched.source_image_samples[1].shape == (1, 6, 5, 3)

    def test_batched_mask_images_preserve_restore_metadata(self):
        first = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        first.source_image_sizes = [(8, 8)]
        first.source_restore_crop_mode = "none"
        first.preprocess_image_sizes = [(4, 4)]
        second = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        second.source_image_sizes = [(8, 8)]
        second.source_restore_crop_mode = "none"
        second.preprocess_image_sizes = [(4, 4)]

        batched = batch_images([first, second])

        assert batched is not None
        assert batched.source_image_sizes == [(8, 8), (8, 8)]
        assert batched.source_restore_crop_mode == "none"
        assert batched.preprocess_image_sizes == [(4, 4), (4, 4)]

    def test_batch_images_pads_to_full_channel_difference(self):
        first = torch.zeros((1, 4, 4, 1), dtype=torch.float32)
        second = torch.ones((1, 4, 4, 4), dtype=torch.float32)

        batched = batch_images([first, second])

        assert batched is not None
        assert batched.shape == (2, 4, 4, 4)
        assert torch.all(batched[0, :, :, 1:] == 1.0)

    def test_image_from_batch_preserves_source_samples_for_mixed_restore(self):
        first = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        second = torch.ones((1, 6, 5, 3), dtype=torch.float32)

        batched = batch_images([first, second])
        selected = ImageFromBatch.execute(batched, 1, 1)[0]

        assert selected.shape == (1, 4, 4, 3)
        assert selected.source_image_sizes == [(6, 5)]
        assert len(selected.source_image_samples) == 1
        assert selected.source_image_samples[0].shape == (1, 6, 5, 3)

    def test_rebatched_mixed_mask_images_restore_after_split(self):
        first_mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        first_mask.source_image_sizes = [(6, 6)]
        first_mask.source_restore_crop_mode = "none"
        first_mask.preprocess_image_sizes = [(4, 4)]
        second_mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        second_mask.source_image_sizes = [(2, 3)]
        second_mask.source_restore_crop_mode = "none"
        second_mask.preprocess_image_sizes = [(4, 4)]

        first_image = MaskToImage.execute(first_mask)[0]
        second_image = MaskToImage.execute(second_mask)[0]
        batched = batch_images([first_image, second_image])

        def fake_upscale(sample, width, height, method, crop):
            assert sample.shape == (1, 3, 4, 4)
            assert (width, height, method, crop) == (3, 2, "bilinear", "disabled")
            return torch.full((sample.shape[0], sample.shape[1], height, width), 0.5, dtype=sample.dtype)

        with patch.object(nodes_images.comfy.utils, "common_upscale", side_effect=fake_upscale) as common_upscale:
            result = ImageFromBatch.execute(batched, 1, 1)[0]

        assert result.shape == (1, 2, 3, 3)
        assert common_upscale.call_count == 1

    def test_batch_images_backfills_missing_source_samples_before_late_metadata(self):
        first = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        first.source_image_sizes = [(4, 4)]
        second = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        second.source_image_sizes = [(6, 5)]
        second.source_image_samples = [torch.ones((1, 6, 5, 3), dtype=torch.float32)]

        batched = batch_images([first, second])

        assert batched is not None
        assert batched.source_image_sizes == [(4, 4), (6, 5)]
        assert len(batched.source_image_samples) == 2
        assert batched.source_image_samples[0].shape == (1, 4, 4, 3)
        assert batched.source_image_samples[1].shape == (1, 6, 5, 3)

    def test_metadata_bearing_sample_keeps_restore_mode_in_mixed_batch(self):
        restored = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        restored.source_image_sizes = [(6, 5)]
        restored.source_restore_crop_mode = "none"
        restored.preprocess_image_sizes = [(4, 4)]
        plain = torch.ones((1, 4, 4, 3), dtype=torch.float32)

        batched = batch_images([restored, plain])

        assert batched is not None
        assert batched.source_restore_crop_mode == ["none", None]
        assert batched.preprocess_image_sizes == [(4, 4), None]

        def fake_upscale(sample, width, height, method, crop):
            assert sample.shape == (1, 3, 4, 4)
            assert (width, height, method, crop) == (5, 6, "bilinear", "disabled")
            return torch.full((sample.shape[0], sample.shape[1], height, width), 0.5, dtype=sample.dtype)

        with patch.object(nodes_images.comfy.utils, "common_upscale", side_effect=fake_upscale) as common_upscale:
            result = ImageFromBatch.execute(batched, 0, 1)[0]

        assert result.shape == (1, 6, 5, 3)
        assert result.source_restore_crop_mode == "none"
        assert result.preprocess_image_sizes == [(4, 4)]
        assert common_upscale.call_count == 1

    def test_mixed_batch_preserves_optional_preprocess_sizes(self):
        restored = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        restored.preprocess_image_sizes = [(4, 4)]
        plain = torch.ones((1, 4, 4, 3), dtype=torch.float32)

        batched = batch_images([restored, plain])
        sliced = ImageFromBatch.execute(batched, 0, 2)[0]

        assert batched.preprocess_image_sizes == [(4, 4), None]
        assert sliced.preprocess_image_sizes == [(4, 4), None]

    def test_batch_images_flattens_per_sample_crop_mode_lists(self):
        mixed = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        mixed.source_image_sizes = [(6, 5), (4, 4)]
        mixed.source_restore_crop_mode = ["none", None]
        mixed.preprocess_image_sizes = [(4, 4), (4, 4)]

        rebatched = batch_images([mixed])

        assert rebatched is not None
        assert rebatched.source_restore_crop_mode == ["none", None]

    def test_image_from_batch_preserves_mixed_crop_mode_list_for_reslicing(self):
        mixed = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
        mixed.source_image_sizes = [(6, 5), (4, 4)]
        mixed.source_restore_crop_mode = ["none", None]
        mixed.preprocess_image_sizes = [(4, 4), (4, 4)]

        sliced = ImageFromBatch.execute(mixed, 0, 2)[0]

        assert sliced.source_restore_crop_mode == ["none", None]

        def fake_upscale(sample, width, height, method, crop):
            assert sample.shape == (1, 3, 4, 4)
            assert (width, height, method, crop) == (5, 6, "bilinear", "disabled")
            return torch.full((sample.shape[0], sample.shape[1], height, width), 0.5, dtype=sample.dtype)

        with patch.object(nodes_images.comfy.utils, "common_upscale", side_effect=fake_upscale) as common_upscale:
            restored = ImageFromBatch.execute(sliced, 0, 1)[0]

        assert restored.shape == (1, 6, 5, 3)
        assert restored.source_restore_crop_mode == "none"
        assert common_upscale.call_count == 1
