"""
demo_infer.py — End-to-end inference demo: CFAR candidate proposal -> CNN
scoring -> labeled detections drawn on the image + a JSON report.

USAGE:
  python3 demo_infer.py --image test_sonar.png --model ghostnet_scorer.pt \
      --out annotated.png --report detections.json
"""


import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


from cfar import CFARConfig, propose
from train_cnn import GhostNetScorer, CLASS_LIST, IMG_SIZE

# colors per class for the drawn boxes (BGR)
BOX_COLORS = {
    "net": (0, 0, 255), "bottle": (0, 165, 255), "pipe": (0, 255, 255),
    "cylinder": (255, 0, 255), "wreck": (0, 255, 0), "aircraft": (255, 255, 0),
    "human": (255, 0, 0),
}


def load_model(model_path, device):
    model = GhostNetScorer(len(CLASS_LIST)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def crops_to_batch(img, boxes):
    """Crop every CFAR candidate box, resize to IMG_SIZE, stack into one
    tensor -- batched inference is far faster than one forward pass per
    candidate when CFAR proposes thousands of boxes."""
    tensors = []
    valid_boxes = []
    for (x, y, w, h) in boxes:
        crop = img[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        tensors.append(crop)
        valid_boxes.append((x, y, w, h))
    if not tensors:
        return None, []
    batch = np.stack(tensors).astype(np.float32) / 255.0
    batch = torch.from_numpy(batch).unsqueeze(1)  # (N, 1, H, W)
    return batch, valid_boxes


def run_inference(img, model, device, cfar_cfg, conf_thresh=0.6, batch_size=256):
    candidates, _ = propose(img, cfar_cfg)
    print(f"[demo] CFAR proposed {len(candidates)} candidates")

    batch, boxes = crops_to_batch(img, candidates)
    if batch is None:
        print("[demo] no valid candidates to score")
        return []

    detections = []
    with torch.no_grad():
        for i in range(0, len(boxes), batch_size):
            chunk = batch[i:i + batch_size].to(device)
            logits = model(chunk)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for j in range(chunk.size(0)):
                cls_idx = pred[j].item()
                cls_name = CLASS_LIST[cls_idx]
                c = conf[j].item()
                if cls_name == "background":
                    continue
                if c < conf_thresh:
                    continue
                x, y, w, h = boxes[i + j]
                detections.append({
                    "class": cls_name, "confidence": round(c, 3),
                    "bbox_xywh": [int(x), int(y), int(w), int(h)],
                })
    return detections


def non_max_suppress_per_class(detections, iou_thresh=0.3, max_per_class=8):
    """CFAR proposes many overlapping candidates around the same real object
    -- without this, one true detection can produce 5-10 near-duplicate
    boxes on the output image. Simple per-class greedy NMS + a display cap
    keeps the demo image readable."""
    by_class = {}
    for d in detections:
        by_class.setdefault(d["class"], []).append(d)

    kept = []
    for cls_name, dets in by_class.items():
        dets = sorted(dets, key=lambda d: -d["confidence"])
        chosen = []
        for d in dets:
            x, y, w, h = d["bbox_xywh"]
            overlaps = False
            for c in chosen:
                cx, cy, cw, ch = c["bbox_xywh"]
                ix1, iy1 = max(x, cx), max(y, cy)
                ix2, iy2 = min(x + w, cx + cw), min(y + h, cy + ch)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                union = w * h + cw * ch - inter
                if union > 0 and inter / union > iou_thresh:
                    overlaps = True
                    break
            if not overlaps:
                chosen.append(d)
            if len(chosen) >= max_per_class:
                break
        kept.extend(chosen)
    return kept


def draw_detections(img, detections):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for d in detections:
        x, y, w, h = d["bbox_xywh"]
        color = BOX_COLORS.get(d["class"], (255, 255, 255))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{d['class']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(vis, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="annotated.png")
    ap.add_argument("--report", default="detections.json")
    ap.add_argument("--conf-thresh", type=float, default=0.6)
    ap.add_argument("--guard-cells", type=int, default=67)
    ap.add_argument("--train-cells", type=int, default=15)
    ap.add_argument("--pfa-scale", type=float, default=2.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] device: {device}")

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not load {args.image}")

    model = load_model(args.model, device)
    cfar_cfg = CFARConfig(guard_cells=args.guard_cells, train_cells=args.train_cells,
                           pfa_scale=args.pfa_scale)

    detections = run_inference(img, model, device, cfar_cfg, conf_thresh=args.conf_thresh)
    print(f"[demo] {len(detections)} raw detections above confidence {args.conf_thresh}")

    detections = non_max_suppress_per_class(detections)
    print(f"[demo] {len(detections)} detections after per-class NMS")
    for d in detections:
        print(f"    {d['class']:>10} conf={d['confidence']:.2f} bbox={d['bbox_xywh']}")

    vis = draw_detections(img, detections)
    cv2.imwrite(args.out, vis)
    print(f"[demo] annotated image -> {args.out}")

    with open(args.report, "w") as f:
        json.dump({"image": args.image, "detections": detections}, f, indent=2)
    print(f"[demo] detections report -> {args.report}")


if __name__ == "__main__":
    main()