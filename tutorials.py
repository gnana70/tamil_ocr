from ocr_tamil.ocr import OCR

image_path = r"test_images\english_1.png" # insert your own path here
ocr = OCR(detect=True,enable_cuda=False)
texts = ocr.predict(image_path)

with open("output.txt","w",encoding="utf-8") as f:
    f.write(texts)