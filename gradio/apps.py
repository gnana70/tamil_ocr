import gradio as gr
from ocr_tamil.ocr import OCR

model_path = r"parseq_tamil_v6.ckpt"
text_detect_model = r"craft_mlt_25k.pth" # add the full path of the craft model
ocr_detect = OCR(detect=True,tamil_model_path=model_path,detect_model_path=text_detect_model,enable_cuda=True)
ocr_recognize = OCR(detect=False,tamil_model_path=model_path,detect_model_path=text_detect_model,enable_cuda=True)

def predict(image_path,mode):
    if mode == "recognize":
        texts = ocr_recognize.predict(image_path)
    else:
        texts = ocr_detect.predict(image_path)
    return texts


image_examples = ["0.jpg","1.jpg","2.jpg","3.jpg","4.jpg","5.jpg",
                  "6.jpg","7.jpg","8.jpg","9.jpg","10.jpg","14.jpg"]

mode_examples = ["detect","recognize","recognize","recognize","recognize","recognize"
                 ,"recognize","recognize","recognize","recognize","recognize","recognize"]

input_1 = gr.Image(type="numpy")
input_2 = gr.Radio(["recognize", "detect"], label="mode", 
                   info="Only Text recognition or need both Text detection + recognition")

examples = [[i,j] for i,j in zip(image_examples,mode_examples)]


gr.Interface(
    predict,
    inputs=[input_1,input_2],
    outputs=gr.TextArea(label="Extracted Text",interactive=False,
                       show_copy_button=True),
    title="OCR TAMIL",
    examples=examples
).launch()

