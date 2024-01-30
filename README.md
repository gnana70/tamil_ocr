# OCR Tamil - Easy, Accurate and Simple to use Tamil OCR

<p align="center">
  <a href="LICENSE">
    <img src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/MIT.svg" alt="LICENSE">
  </a>
  <a href="https://huggingface.co/spaces/GnanaPrasath/ocr_tamil">
    <img src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/huggingface.svg" alt="HuggingSpace">
  </a>
</p>


<div align="center">
  <p>
    <a href="https://github.com/gnana70/tamil_ocr">
    <img width="50%" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/logo_1.gif">
    </a>
  </p>
</div>

 OCR Tamil can help you extract text from signboard, nameplates, storefronts etc., from Natural Scenes with high accuracy. This version of OCR is much more robust to tilted text compared to the Tesseract, Paddle OCR and Easy OCR as they are primarily built to work on the documents texts and not on natural scenes. This model is work in progress, feel free to contribute!!!

Currently supports two languages (English + Tamil). Accuracy of the model can be improved by adjusting the Text detection model as per your requirements. Achieved the accuracy of around **>95%** (98% NED) in validation set

## Comparison between Tesseract OCR and OCR Tamil

 Input Image                                                                |  OCR TAMIL            | Tesseract         | 
|:--------------------------------------------------------------------------:|:--------------------:|:-----------------:|
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/4.jpg">                   | வாழ்கவளமுடன்     |    க்‌ க்கஸாரகளள௮ஊகஎளமுடன்‌    | 
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/10.jpg">                  | ரெடிமேட்ஸ்          |**NO OUTPUT**      | 
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/2.jpg">                   | கோபி               | **NO OUTPUT**            | 
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/6.jpg">                   | தாம்பரம்            | **NO OUTPUT** | 
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/1.jpg">                   | நெடுஞ்சாலைத்      | **NO OUTPUT**             |
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/5.jpg">                   | அண்ணாசாலை      | **NO OUTPUT**             |  

**Obtained Tesseract results using the [huggingface space](https://huggingface.co/spaces/kneelesh48/Tesseract-OCR) with Tamil as language**

## How to Install and Use OCR Tamil 

**Tested using Python 3.10 on Windows & Linux (Ubuntu 22.04) Machines**
### Pip
1. Using PIP install 
```pip install ocr_tamil```
2. Use the below code for text recognition at word level by inserting the image_path

**Text Recognition**
```python
from ocr_tamil.ocr import OCR
image_path = r"test_images\1.jpg" # insert your own path here (step 2 file location)
ocr = OCR()
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
ocr = OCR(detect=True)
texts = ocr.predict(image_path)
with open("output.txt","w",encoding="utf-8") as f:
    f.write(texts)

>>>> கொடைக்கானல் Kodaikanal 

```
<img width="400" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/0.jpg">


### Github
1. Clone the repository
2. Pip install the required modules using ```pip install -r requirements.txt```
3. Run the below code by providing the image path 

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

#### Detailed Medium tutorial can be found [here](https://gnana70.medium.com/ocr-tamil-easy-accurate-and-simple-to-use-tamil-ocr-b03b98697f7b). 

#### Huggingface spaces🤗 demo can be found [here](https://huggingface.co/spaces/GnanaPrasath/ocr_tamil)

## Applications
1. Navigating autonomous vehicles based on the signboards
2. License plate recognition

## Limitations

1. Unable to read the text if they are present in rotated forms

<p align="left">
<img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/9.jpg"> 
<img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/8.jpg">
</p>

2. Currently supports Only English and Tamil Language

3. Document Text reading capability is limited. Auto identification of Paragraph, line are not supported along with Text detection model inability to detect and crop the Tamil text leads to accuracy decrease (**WORKAROUND** Can use your own text detection model along with OCR tamil text recognition model)
<p align="center">
<img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/tamil_sentence.jpg">
</p>
<p align="center">
<span>Cropped Text from Text detection Model</span>
</p>
<p align="center">
<img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/tamil_sentence_crop.jpg">
</p>
<p align="center">
Character **இ** missing due to text detection model error 
</p>

**?**யற்கை மூலிகைகளில் இருந்து ஈர்த்தெடுக்கக்கப்பட்ட விரிய உட்பொருட்களை உள்ளடக்கி எந்த இரசாயன சேர்க்கைகளும் **?**ல்லாமல் உருவாக்கப்பட்ட **?**ந்தியாவின் முதல் சித்த தயாரிப்பு 


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

![logo](https://github.com/gnana70/tamil_ocr/raw/main/test_images/logo_1.gif)