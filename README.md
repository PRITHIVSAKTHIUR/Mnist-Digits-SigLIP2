![fQPjrpOKabPgt_9vCH4Qj.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/rB4X4q0YZkX0WJW6fZ83F.png)
![ssdsdsdfsdfcsdfc.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/cyqhEw4goojpJ2shwDdEb.png)

# **Mnist-Digits-SigLIP2**  

> **Mnist-Digits-SigLIP2** is an image classification model fine-tuned from **google/siglip2-base-patch16-224** to classify handwritten digits (0-9) using the **SiglipForImageClassification** architecture. It is trained on the MNIST dataset for accurate digit recognition.  

```py
Classification Report:
              precision    recall  f1-score   support

           0     0.9988    0.9959    0.9974      5923
           1     0.9987    0.9918    0.9952      6742
           2     0.9918    0.9943    0.9930      5958
           3     0.9975    0.9938    0.9957      6131
           4     0.9892    0.9882    0.9887      5842
           5     0.9859    0.9937    0.9898      5421
           6     0.9936    0.9939    0.9937      5918
           7     0.9856    0.9943    0.9899      6265
           8     0.9932    0.9921    0.9926      5851
           9     0.9926    0.9897    0.9912      5949

    accuracy                         0.9928     60000
   macro avg     0.9927    0.9928    0.9927     60000
weighted avg     0.9928    0.9928    0.9928     60000
```

![download (2).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/qUaioZfL840_BrRhReCqd.png)

### **Classes:**  
- **Class 0:** "0"  
- **Class 1:** "1"  
- **Class 2:** "2"  
- **Class 3:** "3"  
- **Class 4:** "4"  
- **Class 5:** "5"  
- **Class 6:** "6"  
- **Class 7:** "7"  
- **Class 8:** "8"  
- **Class 9:** "9"  

---

# **Run with Transformers🤗**  

```python
!pip install -q transformers torch pillow gradio
```  

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Mnist-Digits-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def classify_digit(image):
    """Predicts the digit in the given handwritten digit image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
        "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=classify_digit,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="MNIST Digit Classification 🔢",
    description="Upload a handwritten digit image (0-9) to recognize it using MNIST-Digits-SigLIP2."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

---

# **Sample Inference**  

![Screenshot 2025-03-28 at 23-23-02 MNIST Digit Classification 🔢.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/o0YinTlr6or3V_wOJMCf3.png)
![Screenshot 2025-03-28 at 23-25-22 MNIST Digit Classification 🔢.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/LP4upkfHfUa3wdRSSS9tp.png)
![Screenshot 2025-03-28 at 23-25-52 MNIST Digit Classification 🔢.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/XJ0AmEg0Com-KN32jtGDu.png)
![Screenshot 2025-03-28 at 23-26-52 MNIST Digit Classification 🔢.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/rboO-rw7BxK7S8vJMF-To.png)

# **Intended Use:**  

The **Mnist-Digits-SigLIP2** model is designed for handwritten digit recognition. Potential applications include:  

- **Optical Character Recognition (OCR):** Digit recognition for various documents.  
- **Banking & Finance:** Automated check processing.  
- **Education & Learning:** AI-powered handwriting assessment.  
- **Embedded Systems:** Handwriting input in smart devices. 
