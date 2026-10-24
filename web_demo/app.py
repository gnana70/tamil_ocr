from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from ocr_tamil.ocr import OCR
from PIL import Image
import io
import os


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Initialize OCR once
ocr = OCR(detect=True)


def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT


@app.route('/', methods=['GET', 'POST'])
def index():
    result_text = None
    if request.method == 'POST':
        # handle file upload
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('index.html', error='No file selected')
        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            return render_template('index.html', error='Unsupported file type')
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Run OCR
        texts = ocr.predict(save_path)
        # ocr.predict returns list of lists; join lines
        if isinstance(texts, list) and len(texts) > 0:
            if isinstance(texts[0], list):
                # detection + recognition -> list of lines
                lines = [" ".join(item) if isinstance(item, list) else str(item) for item in texts[0]]
                result_text = '\n'.join(lines)
            else:
                result_text = '\n'.join(texts)
        else:
            result_text = ''

        return render_template('index.html', result=result_text, filename=filename)

    return render_template('index.html')


@app.route('/download/<filename>')
def download(filename):
    # return extracted text as txt file
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    texts = ocr.predict(path)
    text = ''
    if isinstance(texts, list) and len(texts) > 0:
        if isinstance(texts[0], list):
            lines = [" ".join(item) if isinstance(item, list) else str(item) for item in texts[0]]
            text = '\n'.join(lines)
        else:
            text = '\n'.join(texts)
    buf = io.BytesIO()
    buf.write(text.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"{filename}.txt", mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)