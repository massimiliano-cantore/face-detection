"""Rilevatori di volti: HOG + SVM (classico) e YuNet (rete convoluzionale leggera).

Entrambi restituiscono una lista di tuple (x1, y1, x2, y2, score) in coordinate dell'immagine originale.
"""
import json
import os
import urllib.request

import cv2
import numpy as np

YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def iou(a, b):
    xa, ya, xb, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def nms(dets, thr):
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    keep = []
    while dets:
        cur = dets.pop(0)
        keep.append(cur)
        dets = [d for d in dets if iou(cur, d) < thr]
    return keep


class HogSvmDetector:
    """Finestra scorrevole su piramide di scale, feature HOG, SVM con kernel RBF approssimato (Nystroem)."""

    def __init__(self, model_dir):
        import joblib
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.model = joblib.load(os.path.join(model_dir, "hog_svm_pipeline.joblib"))
        self.patch, self.cell, self.work_w = cfg["PATCH"], cfg["CELL"], cfg["WORK_W"]
        self.scale, self.step, self.nms_iou = cfg["SCALE"], cfg["STEP"], cfg["NMS_IOU"]
        self.threshold = cfg["score_threshold"]
        self.blocks = self.patch // self.cell - 1

    def _windows(self, gray):
        from skimage.feature import hog
        H = hog(gray, orientations=9, pixels_per_cell=(self.cell, self.cell), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=False)
        s, b = self.step // self.cell, self.blocks
        rows, cols = range(0, H.shape[0] - b + 1, s), range(0, H.shape[1] - b + 1, s)
        feats = np.array([H[r:r + b, c:c + b].ravel() for r in rows for c in cols])
        xy = np.array([(c * self.cell, r * self.cell) for r in rows for c in cols])
        return feats, xy

    def detect(self, img_bgr, threshold=None):
        thr = self.threshold if threshold is None else threshold
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h0, w0 = gray.shape
        scale0 = w0 / self.work_w
        img = cv2.resize(gray, (self.work_w, int(round(h0 / scale0))), interpolation=cv2.INTER_AREA)
        dets, ls, P = [], 1.0, self.patch
        while img.shape[0] >= P and img.shape[1] >= P:
            feats, xy = self._windows(img)
            if len(feats):
                sc = self.model.decision_function(feats)
                f = scale0 * ls
                keep = sc >= thr
                dets += [(x * f, y * f, (x + P) * f, (y + P) * f, float(s)) for (x, y), s in zip(xy[keep], sc[keep])]
            ls *= self.scale
            img = cv2.resize(img, (int(img.shape[1] / self.scale), int(img.shape[0] / self.scale)),
                             interpolation=cv2.INTER_AREA)
        return nms(dets, self.nms_iou)


class YuNetDetector:
    """YuNet (Wu et al., 2023) tramite cv2.FaceDetectorYN: ~75k parametri, 230 KB."""

    def __init__(self, model_path="face_detection_yunet_2023mar.onnx", work_w=640, threshold=0.6, nms_iou=0.3):
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(YUNET_URL, model_path)
        self.net = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold=0.3,
                                             nms_threshold=nms_iou, top_k=5000)
        self.work_w, self.threshold = work_w, threshold

    def detect(self, img_bgr, threshold=None):
        thr = self.threshold if threshold is None else threshold
        h0, w0 = img_bgr.shape[:2]
        s = w0 / self.work_w
        img = cv2.resize(img_bgr, (self.work_w, int(round(h0 / s))))
        self.net.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.net.detect(img)
        if faces is None:
            return []
        return [(x * s, y * s, (x + w) * s, (y + h) * s, float(f[-1]))
                for f in faces for x, y, w, h in [f[:4]] if f[-1] >= thr]


def draw(img_bgr, dets, color=(0, 200, 0)):
    vis = img_bgr.copy()
    t = max(2, img_bgr.shape[1] // 300)
    for x1, y1, x2, y2, _ in dets:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, t)
    return vis
