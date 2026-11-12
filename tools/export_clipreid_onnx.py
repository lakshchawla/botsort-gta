"""Convert a CLIP-ReID (ViT-B/16) checkpoint to ONNX for the DeepStream ReID SGIE (nvinfer).

    python3 tools/export_clipreid_onnx.py \
        --ckpt ~/Downloads/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth \
        --out models/reid/msmt17_clipreid_12x12sie_vitb16.onnx [--sie mean|none|<camera id>]

Everything the model needs is read from the checkpoint itself (number of classes, number of SIE cameras,
stride from the position-embedding length), so the same script converts other CLIP-ReID ViT-B/16 checkpoints.

What the exported graph is
  input   "input"  float32 [batch, 3, 256, 128]  RGB, pixels already mapped by (x - 127.5) / 127.5 outside
                                                  the graph (DeepStream does it: offsets/net-scale-factor)
  output  "output" float32 [batch, 1280]         cat([cls feature (768), projected cls feature (512)]) with
                                                  CLIP-ReID's TEST.NECK_FEAT='before'; NOT L2-normalised
                                                  (the tracker normalises: addFeatureNormalization: 1)

SIE (side-information embedding): CLIP-ReID adds a learned per-camera vector to the class token. A single-input
ONNX cannot take a camera id, so one embedding is baked in as a constant:
  mean  (default)  average of all trained camera embeddings - a neutral "unknown camera"
  none             no camera information (what the model sees if cam_label is omitted)
  <int>            a specific trained camera id
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

CLIP_REID_ROOT = "/home/lakshh/workspace/reid/CLIP-ReID"


def infer_from_state_dict(sd: Dict[str, "torch.Tensor"], size: Tuple[int, int] = (256, 128)) -> dict:
    """Read the model hyper-parameters back out of the checkpoint tensors."""
    n_pos = sd["image_encoder.positional_embedding"].shape[0] - 1        # patch tokens (without CLS)
    patch = 16
    h_max, w_max = size[0] // patch, size[1] // patch
    # stride s gives ((H-16)//s+1) * ((W-16)//s+1) tokens: find the stride that matches
    stride = next((s for s in range(1, 17)
                   if ((size[0] - 16) // s + 1) * ((size[1] - 16) // s + 1) == n_pos), None)
    if stride is None:
        raise ValueError(f"no stride in 1..16 yields {n_pos} patch tokens for input {size}")
    return {
        "num_classes": sd["classifier.weight"].shape[0],
        "camera_num": sd["cv_embed"].shape[0] if "cv_embed" in sd else 0,
        "sie_camera": "cv_embed" in sd,
        "stride": stride,
        "feat_dim": sd["bottleneck.weight"].shape[0] + sd["bottleneck_proj.weight"].shape[0],
    }


def build_model(ckpt: str, info: dict, sie_coe: float, clip_reid_root: str = CLIP_REID_ROOT):
    sys.path.insert(0, clip_reid_root)
    from config import cfg                                   # CLIP-ReID repo
    from model.make_model_clipreid import make_model
    cfg.merge_from_file(os.path.join(clip_reid_root, "configs/person/vit_clipreid.yml"))
    cfg.merge_from_list(["MODEL.SIE_CAMERA", info["sie_camera"], "MODEL.SIE_COE", sie_coe,
                         "MODEL.STRIDE_SIZE", [info["stride"], info["stride"]]])
    cfg.freeze()
    model = make_model(cfg, num_class=info["num_classes"], camera_num=max(info["camera_num"], 1), view_num=1)
    model.load_param(ckpt)
    return model.cpu().eval()


def encode(visual, x, cv):
    """CLIP-ReID's VisionTransformer.forward with the camera embedding added to the class token.

    Same maths as model/clip/model.py::VisionTransformer.forward(x, cv_emb); re-written because the
    original guards the SIE term with `if cv_emb != None:` (a data-dependent branch) and edits the
    token tensor in place, neither of which torch.export can trace. Returns (cls feature 768-d,
    projected cls feature 512-d)."""
    import torch
    x = visual.conv1(x)                                            # [B, C, h, w]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)     # [B, h*w, C]
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype)
    if cv is not None:
        cls = cls + cv                                             # SIE: camera vector goes onto the CLS token only
    x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x).permute(1, 0, 2)                          # NLD -> LND
    x = visual.transformer(x).permute(1, 0, 2)                     # all 12 blocks, back to NLD
    x = visual.ln_post(x)
    return x[:, 0], (x @ visual.proj)[:, 0]


def make_wrapper(model, sie: str, sie_coe: float):
    import torch

    class Wrapper(torch.nn.Module):
        """image -> 1280-d feature, with the SIE camera embedding baked in as a constant."""

        def __init__(self):
            super().__init__()
            self.model = model
            if sie == "none" or not hasattr(model, "cv_embed"):
                cv = None
            elif sie == "mean":
                cv = sie_coe * model.cv_embed.detach().mean(dim=0)
            else:
                cv = sie_coe * model.cv_embed.detach()[int(sie)]
            self.cv = cv

        def forward(self, x):
            feat, feat_proj = encode(self.model.image_encoder, x, self.cv)
            return torch.cat([feat, feat_proj], dim=1)

    return Wrapper().eval()


def check_against_repo_forward(wrapper, model, size, atol=1e-4) -> float:
    """The re-written encoder must equal the repo's own forward (eager mode, same SIE vector)."""
    import torch
    x = torch.randn(2, 3, *size)
    with torch.no_grad():
        _, f, fp = model.image_encoder(x, wrapper.cv)
        ref = torch.cat([f[:, 0], fp[:, 0]], dim=1)
        got = wrapper(x)
    diff = float((ref - got).abs().max())
    assert diff < atol, f"re-written encoder differs from the repo forward by {diff}"
    return diff


def export(wrapper, out_path: str, size: Tuple[int, int], max_batch: int = 128) -> None:
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # Dynamo exporter (the legacy tracer cannot handle nn.MultiheadAttention here). Trace with batch 2: a batch
    # of 1 would be specialised away and the dynamic batch dim would not stick.
    dummy = torch.randn(2, 3, *size)
    torch.onnx.export(wrapper, (dummy,), out_path, input_names=["input"], output_names=["output"],
                      dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=max_batch)},),
                      opset_version=18, dynamo=True)


def finalize_for_tensorrt(onnx_path: str) -> dict:
    """Make the exported graph digestible for TensorRT (DeepStream 7.1 / TRT 10.3).

    The dynamo exporter writes every LayerNormalization with three outputs (Y, Mean, InvStdDev); the last
    two are never used, but TensorRT's ONNX parser rejects the node ("emptyOutputChecker"). They are
    dropped here (after checking nothing consumes them), then onnxsim folds the constants / shape
    plumbing the exporter leaves behind (about 2000 -> a few hundred nodes). Weights stay in the
    external <name>.onnx.data file next to the model."""
    import onnx
    m = onnx.load(onnx_path)                                       # pulls the external weights in
    n_before = len(m.graph.node)
    consumed = {i for n in m.graph.node for i in n.input} | {o.name for o in m.graph.output}
    trimmed = 0
    for n in m.graph.node:
        if n.op_type == "LayerNormalization" and len(n.output) > 1:
            assert not any(o in consumed for o in n.output[1:]), "an extra LayerNormalization output is used"
            del n.output[1:]
            trimmed += 1
    try:
        from onnxsim import simplify
        m, ok = simplify(m)
        assert ok, "onnxsim could not validate the simplified model"
    except ImportError:
        print("onnxsim not installed: keeping the unsimplified graph")
    data = os.path.basename(onnx_path) + ".data"
    data_path = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), data)
    if os.path.exists(data_path):
        os.remove(data_path)                                       # rewritten below
    onnx.save_model(m, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data, size_threshold=1024)
    return {"layernorm_outputs_trimmed": trimmed, "nodes": f"{n_before} -> {len(m.graph.node)}"}


def verify(wrapper, onnx_path: str, size: Tuple[int, int], feat_dim: int) -> float:
    import numpy as np
    import onnxruntime as ort
    import torch
    x = torch.randn(3, 3, *size)
    with torch.no_grad():
        ref = wrapper(x).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    assert out.shape == (3, feat_dim), f"unexpected output shape {out.shape}"
    one = sess.run(None, {"input": x[:1].numpy()})[0]                       # dynamic batch really works
    assert np.allclose(one, out[:1], atol=1e-3)
    diff = float(np.abs(ref - out).max())
    assert np.allclose(ref, out, atol=1e-3), f"ONNX diverges from PyTorch (max abs diff {diff})"
    return diff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sie", default="mean", help="mean | none | <camera id>   (see module docstring)")
    ap.add_argument("--sie-coe", type=float, default=1.0,
                    help="SIE_COE the checkpoint was trained with (README: MODEL.SIE_COE 1.0; the library default 3.0 is wrong here)")
    ap.add_argument("--size", type=int, nargs=2, default=(256, 128), metavar=("H", "W"))
    ap.add_argument("--clip-reid-root", default=CLIP_REID_ROOT)
    args = ap.parse_args()

    import torch
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    info = infer_from_state_dict(sd, tuple(args.size))
    print("checkpoint:", info)
    if args.sie not in ("mean", "none") and not (args.sie.isdigit() and int(args.sie) < max(info["camera_num"], 1)):
        sys.exit(f"--sie must be mean, none or a camera id < {info['camera_num']}")

    model = build_model(args.ckpt, info, args.sie_coe, args.clip_reid_root)
    wrapper = make_wrapper(model, args.sie, args.sie_coe)
    print(f"encoder rewrite vs repo forward: max |diff| = {check_against_repo_forward(wrapper, model, tuple(args.size)):.2e}")
    export(wrapper, args.out, tuple(args.size))
    print("tensorrt clean-up:", finalize_for_tensorrt(args.out))
    diff = verify(wrapper, args.out, tuple(args.size), info["feat_dim"])
    print(f"exported {args.out}  (sie={args.sie}, coe={args.sie_coe}, max |torch-onnx| = {diff:.2e})")


if __name__ == "__main__":
    main()
