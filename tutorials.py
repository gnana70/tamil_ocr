from ocr_tamil.ocr import OCR

# For a single image - text recognize
image_path = r"test_images\1.jpg" # insert your own path here
ocr = OCR()
text_list = ocr.predict(image_path)

with open("output_text_recognize_single_image.txt","w",encoding="utf-8") as f:
    for text in text_list:
        f.write(text + " ")

print(text_list[0])

# For multiple image - text recognize
image_path = [r"test_images\1.jpg",r"test_images\2.jpg"] # insert your own path here
ocr = OCR()
text_list = ocr.predict(image_path)

with open("output_text_recognize_multiple_image.txt","w",encoding="utf-8") as f:
    for text in text_list:
        f.write(text + "\n")

for text in text_list:
    print(text)

# For single image - text detect + text recognize
image_path = r"test_images\0.jpg" # insert your own path here
ocr = OCR(detect=True)
text_list = ocr.predict(image_path)

with open("output_text_detect_recognize_single_image.txt","w",encoding="utf-8") as f:
    for text in text_list:
        f.write(text + " ")

print(text_list[0])

# For multiple image - text detect + text recognize
image_path = [r"test_images\0.jpg",r"test_images\tamil_sentence.jpg"] # insert your own path here
ocr = OCR(detect=True)
text_list = ocr.predict(image_path)

with open("output_text_detect_recognize_multiple_image.txt","w",encoding="utf-8") as f:
    for text in text_list:
        f.write(text + "\n")

for text in text_list:
    print(text)

