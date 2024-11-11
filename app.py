from flask import Flask, request, jsonify
import cv2
import pytesseract
import numpy as np
from PIL import Image
import io
import os
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path accordingly

app = Flask(__name__)

@app.route('/extract_text', methods=['POST'])
def extract_text():
    # Get the image file from the request
    file = request.files.get('image')

    if not file:
        return jsonify({'error': 'No image file provided'}), 400

    # Read the image file into a numpy array
    img = Image.open(io.BytesIO(file.read()))
    img = np.array(img)

    # Convert image to grayscale (Tesseract works better on grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: Thresholding for clearer text
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Perform OCR on the processed image
    text = pytesseract.image_to_string(thresh)

    return jsonify({'extracted_text': text.strip()})

if __name__ == '__main__':
    app.run(debug=True)
