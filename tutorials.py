from ocr_tamil.ocr import OCR

image_path = r"test_images\tamil_sentence.jpg" # insert your own path here
model_path = r"ocr_tamil\model_weights\parseq_tamil_v6.ckpt"
text_detect_model = r"ocr_tamil\model_weights\craft_mlt_25k.pth" # add the full path of the craft model
ocr = OCR(detect=True,tamil_model_path=model_path,detect_model_path=text_detect_model,enable_cuda=False)
texts = ocr.predict(image_path)

with open("output.txt","w",encoding="utf-8") as f:
    f.write(texts)