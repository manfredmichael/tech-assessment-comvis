<div align="center">
<img src="https://i.pinimg.com/736x/ab/b0/dd/abb0dd40b441bdca723bc5ac5421f310.jpg" alt="drawing" width="200"/>
  
<br/>
<br/>
<br/>

# Improper Mask Wearing Detection
Technical Assessment Submission
</div>

* Live Demo Interface: https://huggingface.co/spaces/manfredmichael/mask-detection-manfred 
* Solution: https://docs.google.com/document/d/1TdbJ3y4gS5lxr1HoQt-tcgtuxmxhKO0_PO6N5ix6e5c/edit?usp=sharing
* Training Report: https://api.wandb.ai/links/anakbangkit/42t0ym8p

# How to run yourself

### Install
1. `https://github.com/manfredmichael/tech-assessment-comvis`
2. `pip install -r requirements.txt`. Note: You might want to activate your virtual environment first.
3. `streamlit run app.py`

### Inference

Inference on image:
```python pipelines.py --input_type image --path <image_path>```

Inference on video:
```python pipelines.py --input_type video --path <video_path>```