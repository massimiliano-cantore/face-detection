"""Demo Gradio: HOG + SVM contro YuNet sulla stessa foto (Hugging Face Spaces, CPU)."""
import os
import sys
import time

import cv2
import gradio as gr
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from face_detection.detectors import HogSvmDetector, YuNetDetector, draw
except ImportError:  # su Spaces detectors.py è copiato accanto ad app.py
    from detectors import HogSvmDetector, YuNetDetector, draw

HOG_REPO = os.getenv("HOG_REPO", "MassimilianoCantore/face-detection-hog-svm")
hog_det = HogSvmDetector(snapshot_download(HOG_REPO))
yunet = YuNetDetector()


def run(image_rgb):
    if image_rgb is None:
        return None, None, ""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    out = []
    for det, color in [(hog_det, (0, 200, 0)), (yunet, (255, 120, 0))]:
        t0 = time.time()
        d = det.detect(img)
        ms = 1000 * (time.time() - t0)
        out.append((cv2.cvtColor(draw(img, d, color), cv2.COLOR_BGR2RGB), len(d), ms))
    (a, na, ta), (b, nb, tb) = out
    summary = (f"| | HOG + SVM | YuNet |\n|---|---|---|\n| Volti trovati | {na} | {nb} |\n"
               f"| Tempo (CPU) | {ta:.0f} ms | {tb:.0f} ms |")
    return a, b, summary


with gr.Blocks(title="Face detection: HOG + SVM vs YuNet") as demo:
    gr.Markdown(
        "# Face detection: HOG + SVM vs YuNet\n"
        "Carica una foto: la stessa immagine passa da un rilevatore **classico** (HOG + SVM, addestrato da me su WIDER FACE) "
        "e da **YuNet**, una rete convoluzionale da 230 KB. Su 300 immagini di validation di WIDER FACE: "
        "AP 0.263 contro **0.889**, 3,1 s contro **54 ms** per immagine."
    )
    with gr.Row():
        inp = gr.Image(label="Foto", type="numpy")
        with gr.Column():
            btn = gr.Button("Rileva volti", variant="primary")
            summary = gr.Markdown()
    with gr.Row():
        out_hog = gr.Image(label="HOG + SVM")
        out_yunet = gr.Image(label="YuNet")
    btn.click(run, inp, [out_hog, out_yunet, summary])
    gr.Markdown("Le foto caricate non vengono salvate. Il rilevatore HOG impiega qualche secondo su CPU: è parte del confronto.")

if __name__ == "__main__":
    demo.launch()
