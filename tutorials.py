from ocr_tamil.ocr import OCR

ocr = OCR()

# For a single image - text recognize
image_path = r"test_images\1.jpg" # insert your own path here
text_list = ocr.predict(image_path)

print("Single text recognize",text_list)

with open("outputs\output_text_recognize_single_image.txt","w",encoding="utf-8") as f:
    for text in text_list:
        f.write(text + " ")

# For multiple image - text recognize
image_path = [r"test_images\1.jpg",r"test_images\2.jpg",
              r"test_images\3.jpg",r"test_images\4.jpg",
              r"test_images\5.jpg",r"test_images\6.jpg",
              r"test_images\7.jpg",r"test_images\8.jpg",
              r"test_images\9.jpg",r"test_images\10.jpg",
              r"test_images\10_180.jpg",r"test_images\11.jpg",
              r"test_images\14.jpg"] # insert your own path here
text_list = ocr.predict(image_path)

print("Multiple text recognize",text_list)

with open("outputs\output_text_recognize_multiple_image.txt","w",encoding="utf-8") as f:
    for text in text_list:
        f.write(text + "\n")

ocr = OCR(detect=True)
# For single image - text detect + text recognize
image_path = r"test_images\0.jpg" # insert your own path here
text_list = ocr.predict(image_path)

print("Single text detect recognize",text_list)

with open("outputs\output_text_detect_recognize_single_image.txt","w",encoding="utf-8") as f:
     for item in text_list:
        for text in item:
            f.write(text + " ")
        f.write("\n")

ocr = OCR(detect=True,lang=["tamil"])
image_path = r"test_images\tamil_handwritten.jpg" # insert your own path here
text_list = ocr.predict(image_path)

print("Single text detect recognize",text_list)

with open("outputs\handwritten_image.txt","w",encoding="utf-8") as f:
     for item in text_list:
        for text in item:
            f.write(text + " ")
        f.write("\n")

# For multiple image - text detect + text recognize
image_path = [r"test_images\0.jpg",r"test_images\tamil_sentence.jpg",
              r"test_images\tamil_sentence_1.png",
              r"test_images\tamil_handwritten.jpg",
              r"test_images\tamil_handwritten_1.jpg"] # insert your own path here
text_list = ocr.predict(image_path)

print("Multiple text detect recognize",text_list)

with open("outputs\output_text_detect_recognize_multiple_image.txt","w",encoding="utf-8") as f:
    for item in text_list:
        for text in item:
            f.write(text + " ")
        f.write("\n")


## For the details of 1
ocr = OCR(details=1)
# For a single image - text recognize
image_path = r"test_images\1.jpg" # insert your own path here
text_list = ocr.predict(image_path)
print("Single text recognize with confidence",text_list)


# For multiple image - text recognize
image_path = [r"test_images\1.jpg",r"test_images\2.jpg"] # insert your own path here
text_list = ocr.predict(image_path)
print("Multiple text recognize",text_list)

ocr = OCR(detect=True,details=1)
# For single image - text detect + text recognize
image_path = r"test_images\0.jpg" # insert your own path here
text_list = ocr.predict(image_path)

print("Single text detect recognize with conf",text_list)

# For multiple image - text detect + text recognize
image_path = [r"test_images\0.jpg",r"test_images\tamil_sentence.jpg"] # insert your own path here
text_list = ocr.predict(image_path)

print("Multiple text detect recognize with conf",text_list)


## For the details of 2
ocr = OCR(detect=True,details=2)
# For single image - text detect + text recognize
image_path = r"test_images\0.jpg" # insert your own path here
text_list = ocr.predict(image_path)

print("Single text detect recognize with confidence and bbox",text_list)

# For multiple image - text detect + text recognize
image_path = [r"test_images\0.jpg",r"test_images\tamil_sentence.jpg"] # insert your own path here
text_list = ocr.predict(image_path)

print("Multiple text detect recognize with confidence and bbox",text_list)


