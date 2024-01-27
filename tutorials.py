from ocr_tamil.ocr import OCR

image_path = r"C:\Users\gnana\Documents\GitHub\tamil_ocr\test_images\1.jpg" # insert your own path here
model_path = r"C:\Users\gnana\Documents\GitHub\tamil_ocr\ocr_tamil\model_weights\parseq_tamil_v6.ckpt"


ocr = OCR(tamil_model_path=model_path,enable_cuda=False)
texts = ocr.predict(image_path)

print(texts)