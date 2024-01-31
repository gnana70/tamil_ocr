from ocr_tamil.ocr import OCR

image_path = r"test_images\4.jpg" # insert your own path here
ocr = OCR(enable_cuda=False)
texts = ocr.predict(image_path)

with open("output.txt","w",encoding="utf-8") as f:
    f.write(texts)