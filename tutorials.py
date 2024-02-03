from ocr_tamil.ocr import OCR

image_path = r"test_images\1.jpg" # insert your own path here
ocr = OCR(detect=False,enable_cuda=False)
texts = ocr.predict(image_path,save_image=True)

with open("output.txt","w",encoding="utf-8") as f:
    f.write(texts)