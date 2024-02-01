from miscellenous.ocr_onnx import OCR

image_path = r"test_images\0.jpg" # insert your own path here
tamil_model_path = r"ocr_tamil\model_weights\parseq_test_tamil.onnx"
eng_model_path = r"ocr_tamil\model_weights\parseq_test_eng.onnx"
text_detect_model = r"ocr_tamil\model_weights\craft_mlt_25k.pth" # add the full path of the craft model
ocr = OCR(detect=True,tamil_model_path=tamil_model_path,
          eng_model_path=eng_model_path,detect_model_path=text_detect_model,enable_cuda=False)
texts = ocr.predict(image_path)

print(texts)

with open("output.txt","w",encoding="utf-8") as f:
    f.write(texts)