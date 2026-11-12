import importlib.util
import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper

_spec = importlib.util.spec_from_file_location(
    "export_clipreid_onnx", os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "export_clipreid_onnx.py"))
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def fake_sd(n_pos, classes=1041, cams=15, sie=True):
    sd = {"image_encoder.positional_embedding": torch.zeros(n_pos + 1, 8),
          "classifier.weight": torch.zeros(classes, 768),
          "bottleneck.weight": torch.zeros(768), "bottleneck_proj.weight": torch.zeros(512)}
    if sie:
        sd["cv_embed"] = torch.zeros(cams, 768)
    return sd


def test_infer_from_state_dict_reads_stride_classes_cameras():
    info = ex.infer_from_state_dict(fake_sd(21 * 10))            # 12x12 stride at 256x128 -> 21x10 patch tokens
    assert info == {"num_classes": 1041, "camera_num": 15, "sie_camera": True, "stride": 12, "feat_dim": 1280}
    assert ex.infer_from_state_dict(fake_sd(16 * 8, sie=False))["stride"] == 16
    assert ex.infer_from_state_dict(fake_sd(16 * 8, sie=False))["sie_camera"] is False
    with pytest.raises(ValueError):
        ex.infer_from_state_dict(fake_sd(7))                      # no stride gives 7 tokens


def test_finalize_trims_unused_layernorm_outputs_and_keeps_the_model_working(tmp_path):
    """The dynamo exporter writes LayerNormalization with 3 outputs, which TensorRT's parser rejects."""
    scale = helper.make_tensor("s", TensorProto.FLOAT, [4], [1.0, 2.0, 0.5, 1.0])
    bias = helper.make_tensor("b", TensorProto.FLOAT, [4], [0.0, 0.1, 0.0, -0.1])
    ln = helper.make_node("LayerNormalization", ["x", "s", "b"], ["y", "mean_unused", "invstd_unused"], axis=-1, epsilon=1e-5)
    g = helper.make_graph([ln], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])], initializer=[scale, bias])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)   # ORT 1.23 reads IR <= 11
    path = str(tmp_path / "ln.onnx")
    onnx.save(m, path)
    x = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    ref = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]

    report = ex.finalize_for_tensorrt(path)
    assert report["layernorm_outputs_trimmed"] == 1
    out = onnx.load(path, load_external_data=False)
    assert all(len(n.output) == 1 for n in out.graph.node if n.op_type == "LayerNormalization")
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    assert np.allclose(ref, got, atol=1e-6)
    assert ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x[:1]})[0].shape == (1, 4)   # dynamic batch


def test_finalize_refuses_when_an_extra_layernorm_output_is_used(tmp_path):
    ln = helper.make_node("LayerNormalization", ["x", "s"], ["y", "mean"], axis=-1)
    use = helper.make_node("Identity", ["mean"], ["z"])
    s = helper.make_tensor("s", TensorProto.FLOAT, [4], [1, 1, 1, 1])
    g = helper.make_graph([ln, use], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4]),
                           helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 1])], initializer=[s])
    path = str(tmp_path / "ln_used.onnx")
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=9), path)
    with pytest.raises(AssertionError):
        ex.finalize_for_tensorrt(path)
