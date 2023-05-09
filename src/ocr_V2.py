import easyocr
import torch
from src.utils.utilities import preproctext
import os

# Doing OCR. Get bounding boxes.


def gocr(path,image_list):
    ocr_l = []
    if not torch.cuda.is_available():
        # Model loading...Can be shifted to the main script
        reader = easyocr.Reader(['en'])
        for img in image_list:
            bounds = reader.readtext(os.path.join(path,img))
            ocr = ' '.join([i[-2] for i in bounds])
            ocr_l.append(ocr)
    else:
        reader = easyocr.Reader(['en'], cudnn_benchmark=True)
        img_list = [os.path.join(path,file) for file in image_list]
        batch_out = reader.readtext_batched(img_list, n_width=800, n_height=600)
        for ind, bound in enumerate(batch_out):
            ocr = ' '.join([i[-2] for i in bound])
            ocr_l.append(ocr)
    ocr_alph_l, oce_preproc = preproctext(ocr_l)
    return ocr_alph_l, oce_preproc



if __name__=="__main__":
    pth = '/home/azureuser/memes/memeExplain/static/uploads'
    check_img = 'memes_1563.jpg'
    img_list = [check_img]
    out_list = gocr(pth, img_list)
    print(out_list)