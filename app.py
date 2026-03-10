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
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB


def images_to_pdf_bytes(image_bytes_list):
    converted = []
    for raw in image_bytes_list:
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            converted.append(buf.getvalue())
            del img, buf
        except Exception as e:
            logger.warning(f"Skipping unreadable image: {e}")
        gc.collect()

    pdf_buf = io.BytesIO()
    img2pdf.convert(*converted, outputstream=pdf_buf)
    pdf_buf.seek(0)
    return pdf_buf.read()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    files = request.files.getlist("images")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "Không có ảnh nào được gửi lên."}), 400

    image_bytes_list = []
    for f in files:
        if f and f.filename:
            data = f.read()
            try:
                Image.open(io.BytesIO(data)).verify()
                image_bytes_list.append(data)
            except Exception:
                logger.warning(f"Skipping invalid image: {f.filename}")

    if not image_bytes_list:
        return jsonify({"error": "Không tìm thấy ảnh hợp lệ (JPG/PNG)."}), 400

    try:
        pdf_bytes = images_to_pdf_bytes(image_bytes_list)
        del image_bytes_list
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
