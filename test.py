from ocr_tamil.ocr import OCR

ocr = OCR(detect=True,details=2,text_threshold=0.3)
# For single image - text detect + text recognize
image_path = r"test_images\tamil_newspaper.jpg" # insert your own path here
text_list = ocr.predict(image_path)

print("Single text detect recognize",text_list)

with open("outputs\english_tamil.txt","w",encoding="utf-8") as f:
    for item in text_list:
        current_line = 1
        for info in item:
            text,conf,bbox = info
            line = bbox[1]
            if line == current_line:
                f.write(text + " ")
            else:
                f.write("\n"+text+ " ")
                current_line = line

        f.write("\n")