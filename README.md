# OCR Tamil

<p align="center">
  <a href="LICENSE">
    <img src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/MIT.svg" alt="LICENSE">
  </a>
</p>


<div align="center">
  <p>
    <a href="https://github.com/gnana70/tamil_ocr">
    <img width="50%" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/logo.gif">
    </a>
  </p>

</div>

Finetuned version of PARSEQ model on Tamil text. This version of OCR is much more robust to tilted text compared to the Tesseract, Paddle OCR and Easy OCR. This model is work in progress, feel free to contribute!!!

Currently supports two languages (English + Tamil). Accuracy of the model can be improved by adjusting the Text detection model as per your requirements. Achieved the accuracy of around **>95%** (98% NED) in validation set

## OUTPUT

 Input Image                                                                |  OCR TAMIL            | Tesseract         | 
|:--------------------------------------------------------------------------:|:--------------------:|:-----------------:|
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/4.jpg">                   | வாழ்கவளமுடன்     |    க்‌ க்கஸாரகளள௮ஊகஎளமுடன்‌    | 
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/10.jpg">                  | ரெடிமேட்ஸ்          |**NO OUTPUT**      | 
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/2.jpg">                   | கோபி               | **NO OUTPUT**            | 
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/6.jpg">                   | தாம்பரம்            | **NO OUTPUT** | 
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/1.jpg">                   | நெடுஞ்சாலைத்      | **NO OUTPUT**             | 

**Obtained Tesseract results using the [huggingface space](https://huggingface.co/spaces/kneelesh48/Tesseract-OCR) with Tamil as language**

## How to USE this repo

**Tested using Python 3.10 on Windows Machine**
### Pip
1. Using PIP install 
```pip install ocr_tamil```
2. Download the model weights from from the [GDRIVE](https://drive.google.com/drive/folders/1oMxdp7VE4Z0uHQkHr1VIrXYfyjZ_WwFV?usp=sharing) and keep it in the local folder to use in step 3
3. Use the below code for text recognition at word level by inserting the image_path and model path

**Text Recognition**
```python
from ocr_tamil.ocr import OCR
image_path = r"test_images\1.jpg" # insert your own path here (step 2 file location)
model_path = r"parseq_tamil_v6.ckpt" # add the full path of the model(step 2 file location)
ocr = OCR(tamil_model_path=model_path)
texts = ocr.predict(image_path)
with open("output.txt","w",encoding="utf-8") as f:
    f.write(texts)
>>>> நெடுஞ்சாலைத்
```
<img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/1_180.jpg">


**Text Detect + Recognition**

4. Use the below code for text detection and recognition by inserting the image_path and model path s (both detection and recognition models)

```python
from ocr_tamil.ocr import OCR
image_path = r"test_images\0.jpg" # insert your own path here
model_path = r"parseq_tamil_v6.ckpt" # add the full path of the parseq model
text_detect_model = "craft_mlt_25k.pth" # add the full path of the craft model
ocr = OCR(detect=True,tamil_model_path=model_path,detect_model_path=text_detect_model)
texts = ocr.predict(image_path)
with open("output.txt","w",encoding="utf-8") as f:
    f.write(texts)

>>>> கொடைக்கானல் Kodaikanal 

```
<img width="400" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/0.jpg">


### Github
1. Clone the repository
2. Pip install the required modules using ```pip install -r requirements.txt```
3. Download the models weights from the [GDRIVE](https://drive.google.com/drive/folders/1oMxdp7VE4Z0uHQkHr1VIrXYfyjZ_WwFV?usp=sharing) and keep it under model_weights 
    
        |___model_weights
            |_____craft_mlt_25k.pth
            |_____parseq_tamil_v6.ckpt
    
4. Run the below code by providing the path 

**Text Recognition**

```python
from ocr_tamil.ocr import OCR

image_path = r"test_images\1.jpg" # insert your own path here
ocr = OCR()
texts = ocr.predict(image_path)
with open("output.txt","w",encoding="utf-8") as f:
    f.write(texts)

>>>> நெடுஞ்சாலைத்

```

**Text Detect + Recognition**

```python
from ocr_tamil.ocr import OCR

image_path = r"test_images\0.jpg" # insert your own path here
ocr = OCR(detect=True)
texts = ocr.predict(image_path)
with open("output.txt","w",encoding="utf-8") as f:
    f.write(texts)

>>>> கொடைக்கானல் Kodaikanal 

```

## LIMITATIONS

Unable to read the text if they are present in rotated forms

<img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/8.jpg"> 
<img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/9.jpg">


## Thanks to the below contibuters for making awesome Text detection and text recognition models

**Text detection** - [CRAFT TEXT DECTECTION](https://github.com/clovaai/CRAFT-pytorch)

**Text recognition** - [PARSEQ](https://github.com/baudm/parseq)


```bibtex
@InProceedings{bautista2022parseq,
  title={Scene Text Recognition with Permuted Autoregressive Sequence Models},
  author={Bautista, Darwin and Atienza, Rowel},
  booktitle={European Conference on Computer Vision},
  pages={178--196},
  month={10},
  year={2022},
  publisher={Springer Nature Switzerland},
  address={Cham},
  doi={10.1007/978-3-031-19815-1_11},
  url={https://doi.org/10.1007/978-3-031-19815-1_11}
}
```

```bibtex
@inproceedings{baek2019character,
  title={Character Region Awareness for Text Detection},
  author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9365--9374},
  year={2019}
}
```

## CITATION

```bibtex
@InProceedings{GnanaPrasath,
  title={Tamil OCR},
  author={Gnana Prasath D},
  month={01},
  year={2024},
  url={https://github.com/gnana70/tamil_ocr}
}
```