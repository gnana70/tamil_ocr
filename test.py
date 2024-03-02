from ocr_tamil.ocr import OCR

ocr = OCR(detect=True)
# For single image - text detect + text recognize
image_path = r"test_images\english_tamil_1.jpg" # insert your own path here
text_list = ocr.predict(image_path)

print("Single text detect recognize",text_list)

with open("outputs\english_tamil.txt","w",encoding="utf-8") as f:
     for item in text_list:
        for text in item:
            f.write(text + " ")
        f.write("\n")