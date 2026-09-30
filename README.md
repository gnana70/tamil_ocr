<a href="https://youtu.be/5HRN41h-oxs">
  <img src="https://img.youtube.com/vi/5HRN41h-oxs/maxresdefault.jpg" alt="Watch the OCR Tamil video on YouTube" width="100%">
</a>

<h1 align="center"> OCR Tamil - Easy, Accurate and Simple to use Tamil OCR - (ஒளி எழுத்துணரி)</h1>

<p align="center">❤️️❤️️Please star✨ it if you like❤️️❤️️</p>

<p align="center">
  <a href="LICENSE">
    <img src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/MIT.svg" alt="LICENSE">
  </a>
  <a href="https://huggingface.co/spaces/GnanaPrasath/ocr_tamil">
    <img src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/huggingface.svg" alt="HuggingSpace">
  </a>
   <a href="https://colab.research.google.com/drive/11QPPj3EmpoIqnpuIznKeP1icxvVOjfux?usp=sharing">
    <img src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/colab.svg" alt="colab">
  </a>
</p>


<div align="center">
  <p>
    <a href="https://github.com/gnana70/tamil_ocr">
    <img width="50%" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/logo_1.gif">
    </a>
  </p>
</div>

 OCR Tamil can help you extract text from signboard, nameplates, storefronts etc., from Natural Scenes with high accuracy. This version of OCR is much more robust to tilted text compared to the Tesseract, Paddle OCR and Easy OCR as they are primarily built to work on the documents texts and not on natural scenes.

## Languages Supported 🔛
**➡️ English**

**➡️ Tamil (தமிழ்)**

## Accuracy 🎯
✔️ English > 98%

✔️ Tamil > 95%

## Model checkpoint
<a href="https://huggingface.co/GnanaPrasath/ocr_tamil">
🤗 Tamil OCR
</a>


## Comparison between Tesseract OCR, EasyOCR and OCR Tamil ⚖️

🏎️ *10-40% faster inference time than EasyOCR and Tesseract*

 Input Image                                                                |  OCR TAMIL   🏆         | Tesseract         | EasyOCR |
|:--------------------------------------------------------------------------:|:--------------------:|:-----------------:|:-----------------:|
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/4.jpg">                   | வாழ்கவளமுடன்✅     |    க்‌ க்கஸாரகளள௮ஊகஎளமுடன்‌ ❌  | வாழக வளமுடன்❌|
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/11.jpg">                   | தமிழ்வாழ்க✅      | **NO OUTPUT** ❌           | தமிழ்வாழ்க✅      |
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/2.jpg">                   | கோபி ✅              | **NO OUTPUT** ❌          | ப99❌          |
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/6.jpg">                   | தாம்பரம் ✅           | **NO OUTPUT** ❌ | தாம்பரம❌ |
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/1.jpg">                   | நெடுஞ்சாலைத் ✅      | **NO OUTPUT** ❌             |நெடுஞ்சாலைத் ✅      |
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/5.jpg">                   | அண்ணாசாலை ✅     | **NO OUTPUT** ❌            |  ல@I9❌            |
| <img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/10.jpg">                  | ரெடிமேட்ஸ் ✅         |**NO OUTPUT** ❌     | ரெடிமேடஸ் ❌         |

**Obtained Tesseract and EasyOCR results using the [Colab notebook](https://colab.research.google.com/drive/1ylZm6afur85Pe6I10N2_tzuBFl2VIxkW?usp=sharing) with Tamil and english as language**

## Handwritten Text (Experimental)🧪
<img width="500" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/develop/test_images/tamil_handwritten.jpg">


```
MODEL OUTPUT: நிமிர்ந்த நன்னடை மேற்கொண்ட பார்வையும் 
நிலத்தில் யார்க் கும் அஞ்சாத நெறிகளும் 
திமிர்ந்த ஞானச் செருக்கும் இருப்பதால் 
செம்மை மாதர் திறம்புவ தில்லையாம் 
அமிழ்ந்து பேரிரு ளாமறி யாமையில் 
அவல மெய்திக் கலையின்  வாழ்வதை 
உமிழ்ந்து தள்ளுதல் பெண்ணற மாகுமாம் 
உதய கன்ன உரைப்பது கேட்டிரோ 
பாரதியார் 
ஹேமந்த் ம 
```


## How to Install and Use OCR Tamil 👨🏼‍💻

### Quick links🌐
📔 Detailed explanation on [Medium article](https://gnana70.medium.com/ocr-tamil-easy-accurate-and-simple-to-use-tamil-ocr-b03b98697f7b). 

✍️ Experiment in [Colab notebook](https://colab.research.google.com/drive/11QPPj3EmpoIqnpuIznKeP1icxvVOjfux?usp=sharing)

🤗 Test it in [Huggingface spaces](https://huggingface.co/spaces/GnanaPrasath/ocr_tamil)

🎥 Watch the [video on YouTube](https://youtu.be/5HRN41h-oxs)


### Pip install instructions🐍
In your command line, run the following command ```pip install ocr_tamil```

If you are using jupyter notebook , install like ```!pip install ocr_tamil```

### Python Usage - Single image inference

**Text Recognition only**

```python
from ocr_tamil.ocr import OCR

image_path = r"test_images\1.jpg" # insert your own path here
ocr = OCR()
text_list = ocr.predict(image_path)
print(text_list[0])

## OUTPUT : நெடுஞ்சாலைத்
```
<img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/1_180.jpg">


**Text Detect + Recognition**

```python
from ocr_tamil.ocr import OCR

image_path = r"test_images\0.jpg" # insert your own image path here
ocr = OCR(detect=True)
texts = ocr.predict(image_path)
print(" ".join(texts[0]))

## OUTPUT : கொடைக்கானல் Kodaikanal 

```
<img width="400" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/0.jpg">

**For line by line inference Text detect + Recognition**

```
from ocr_tamil.ocr import OCR

def line_print(prediction):
  current_line = 1
  extracted_text = ""
  for text in prediction:
    pred_text = text[0]
    line_details = text[2][1]

    if line_details != current_line:
      extracted_text += "\n" + pred_text + " "
      current_line = line_details
    else:
      extracted_text += pred_text + " "
  return extracted_text


ocr = OCR(detect=True,details=2,batch_size=128)
text_list = ocr.predict([image_path])

print(line_print(text_list[0]))

```


### Batch inference mode 💻

**Text Recognition only**

```python
from ocr_tamil.ocr import OCR

image_path = [r"test_images\1.jpg",r"test_images\2.jpg"] # insert your own image paths here
ocr = OCR()
text_list = ocr.predict(image_path)

for text in text_list:
    print(text)

## OUTPUT : நெடுஞ்சாலைத்
## OUTPUT : கோபி

```

**Text Detect + Recognition**

```python
from ocr_tamil.ocr import OCR

image_path = [r"test_images\0.jpg",r"test_images\tamil_sentence.jpg"] # insert your own image paths here
ocr = OCR(detect=True)
text_list = ocr.predict(image_path)

for item in text_list:
  print(" ".join(item))
    

## OUTPUT : கொடைக்கானல் Kodaikanal 
## OUTPUT : செரியர் யற்கை மூலிகைகளில் இருந்து ஈர்த்தெடுக்க்கப்பட்ட வீரிய உட்பொருட்களை உள்ளடக்கி எந்த இரசாயன சேர்க்கைகளும் இல்லாமல் உருவாக்கப்பட்ட இந்தியாவின் முதல் சித்த தயாரிப்பு 

```

### Advanced usage🚀

OCR module can be initialized by setting following parameters as per your requirements

```
1. Confidence of word ->  OCR(details=1)
2. Bounding Box and Confidence of word -> OCR(detect=True,details=2)
3. To change the CRAFT Text detection settings -> OCR(detect=True,text_threshold=0.5,
                                               link_threshold=0.1,
                                               low_text=0.30)
4. To increase the Batch size of text recognition -> OCR(batch_size=16) # set as per available memory
5. To configure the language to be extracted -> OCR(lang=["tamil"]) # list can take "english" or "tamil" or both. Defaults to both language
```

**Tested using Python 3.12 on Windows & Linux (Ubuntu 22.04) Machines**

## Dataset Preparation
<a href="https://colab.research.google.com/drive/1U6flb8QhrF2psyX7k-Appr69qpclPWkQ?usp=sharing">
Sample code for image generation
</a> 

## Applications⚡
1. ADAS system navigation based on the signboards + maps (hybrid approach) 🚁
2. License plate recognition 🚘

## Limitations⛔

1. Document text reading capability is not supported as library doesn't have

      **➡️Auto identification of Paragraph**

      **➡️Orientation detection**

      **➡️Skew correction**

      **➡️Reading order prediction**

      **➡️Document unwarping**

      **➡️Optimal Text detection for Document text not available**  

      (**WORKAROUND** Bring your own models for above cases and use with OCR tamil for text recognition)


2. Unable to read the text if they are present in rotated forms

<p align="left">
<img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/9.jpg"> 
<img width="200" alt="teaser" src="https://github.com/gnana70/tamil_ocr/raw/main/test_images/8.jpg">
</p>

3. Currently supports Only Tamil Language. I don't own english model as it's taken from open source implementation of parseq


## Acknowledgements 👏

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

## Citation

```bibtex
@InProceedings{GnanaPrasath,
  title={Tamil OCR},
  author={Gnana Prasath D},
  month={01},
  year={2024},
  url={https://github.com/gnana70/tamil_ocr}
}
```
