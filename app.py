import os
import gc
import io
import logging
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
import img2pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max upload

# ── AI model (lazy-loaded to save RAM on startup) ──────────────────────────
_model = None
_processor = None

def get_clip_model():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        logger.info("Loading CLIP model …")
        _model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32,
        )
        _model.eval()
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("CLIP model ready.")
    return _model, _processor


def score_image_against_labels(pil_img, labels: list[str]) -> list[float]:
    """Return similarity score of *pil_img* against each label (0-1 range)."""
    import torch
    model, processor = get_clip_model()
    inputs = processor(text=labels, images=pil_img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  # shape: (1, num_labels)
        probs = logits.softmax(dim=-1).squeeze().tolist()
    # Release intermediate tensors
    del inputs, outputs, logits
    gc.collect()
    if isinstance(probs, float):
        probs = [probs]
    return probs


def sort_images_by_labels(image_bytes_list: list[bytes], labels: list[str]) -> list[int]:
    """
    For each image, find the label it matches best.
    Then order images by the index of that best-matching label.
    Images with the same best label are kept in upload order (stable sort).
    Returns a list of indices representing the sorted order.
    """
    if not labels:
        return list(range(len(image_bytes_list)))

    label_assignments = []
    for idx, img_bytes in enumerate(image_bytes_list):
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            scores = score_image_against_labels(pil_img, labels)
            best_label_idx = scores.index(max(scores))
            label_assignments.append((idx, best_label_idx))
            logger.info(f"Image {idx}: best label = '{labels[best_label_idx]}' (score={max(scores):.3f})")
        except Exception as e:
            logger.warning(f"Failed to score image {idx}: {e}")
            label_assignments.append((idx, len(labels)))  # put at end
        finally:
            gc.collect()

    # Unload model after all inference to free RAM
    global _model, _processor
    _model = None
    _processor = None
    gc.collect()

    label_assignments.sort(key=lambda x: x[1])
    return [idx for idx, _ in label_assignments]


def images_to_pdf_bytes(image_bytes_list: list[bytes]) -> bytes:
    """Convert a list of image byte-strings to a single PDF using img2pdf (stream-based)."""
    pdf_buf = io.BytesIO()
    # img2pdf expects file-like objects or bytes; we pass bytes directly
    img2pdf.convert(*image_bytes_list, outputstream=pdf_buf)
    pdf_buf.seek(0)
    return pdf_buf.read()


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    files = request.files.getlist("images")
    labels_raw = request.form.get("labels", "").strip()

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "Không có ảnh nào được gửi lên."}), 400

    # Read all images into memory as bytes
    image_bytes_list = []
    for f in files:
        if f and f.filename:
            data = f.read()
            # Quick validation – check it's an image
            try:
                Image.open(io.BytesIO(data)).verify()
                image_bytes_list.append(data)
            except Exception:
                logger.warning(f"Skipping invalid image: {f.filename}")

    if not image_bytes_list:
        return jsonify({"error": "Không tìm thấy ảnh hợp lệ."}), 400

    labels = [l.strip() for l in labels_raw.split(",") if l.strip()] if labels_raw else []

    try:
        if labels:
            sorted_indices = sort_images_by_labels(image_bytes_list, labels)
            sorted_bytes = [image_bytes_list[i] for i in sorted_indices]
        else:
            sorted_bytes = image_bytes_list

        pdf_bytes = images_to_pdf_bytes(sorted_bytes)

        # Free memory ASAP
        del image_bytes_list, sorted_bytes
        gc.collect()

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="output.pdf",
        )
    except Exception as e:
        logger.error(f"Conversion error: {e}", exc_info=True)
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
