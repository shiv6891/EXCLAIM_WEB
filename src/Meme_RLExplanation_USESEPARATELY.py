#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from torch import optim, nn
# from torchvision import models, transforms
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from PIL import Image

# import cv2
import time
from tqdm.auto import tqdm
from tqdm import trange 
# from tqdm import trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import config
# from tabulate import tabulate
import sys
import glob
import gc
import math
import os
CUDA_LAUNCH_BLOCKING=1
#torch.cuda.set_device(1)
from transformers.utils import logging
logging.set_verbosity(40)

from transformers import T5Tokenizer, T5ForConditionalGeneration
# --T5
T5tokenizer = T5Tokenizer.from_pretrained("t5-large")

#caption generation
from src.caption_module import generateCaption
from src.ocr import generateOCR
# Default dataset files
img_path = None #Set your test set image folder path here
data_dir = config.DATA_PATH
test_file = 'hvvexp_test.csv'
test_df = pd.read_csv(os.path.join(data_dir, test_file), index_col=0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
data_time = AverageMeter('Data', ':6.3f')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[11]:


# T5 decoder
class HarmemeMemesDatasetAug(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path,
        img_dir,
        mode=None,
#         image_transform,
#         text_transform,
        balance=False,
        dev_limit=None,
        random_state=0,
    ):
        self.mode = mode
        self.samples_frame = pd.read_csv(
                data_path, index_col=0
            )
        self.samples_frame = self.samples_frame.reset_index(
            drop=True
        )
        self.samples_frame.image = self.samples_frame.apply(
            lambda row: (img_dir + '/' + row.image), axis=1
        )

    def __len__(self):
        """This method is called when you do len(instance) 
        for an instance of this class.
        """
        return len(self.samples_frame)

    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.samples_frame.loc[idx, "image"]  
        #print("image name=======>{}".format(img_name))
        ocr = self.samples_frame.loc[idx, "OCR"]
        #ocr = generateOCR(img_name)
        #print("ocr generate for image {} is ======>{}".format(img_name,ocr))
        ent = self.samples_frame.loc[idx, "entity"]
        role = self.samples_frame.loc[idx, "role"]
        caption = self.samples_frame.loc[idx, "caption"]
        # caption = generateCaption(Image.open(img_name))
        #print("caption generated for image {} is =====> {}".format(img_name,caption))

        # ---------------------------------------------
        # T5 douple scenario: prompt input + caption
        T5_source1 = "Generation explanation for "+ent+" as "+role+": "+ocr.replace('\n', ' ').replace(' .', '.')
        T5_source2 = caption

        sample = {
            # "id": img_id, 
            "img_name": img_name,                
            "T5_source1": T5_source1,
            "T5_source2": T5_source2,
            "ent": ent,
            "role": role,
            "caption": caption
            
        }
        return sample

BS = 4 #at least 10 can be tried (12327MiB being used)
hm_dataset_test = HarmemeMemesDatasetAug(config.DATA_PATH, img_path, mode = 'test')
dataloader_test = DataLoader(hm_dataset_test, batch_size=BS,
                        shuffle=False, num_workers=0)


# In[ ]:


# For T5 based model
def infer_model(model):
    model.eval()
    code_prof = False
    generated_result = []
    img_list = []

    with torch.no_grad():
        # for data in dataloader_test:  
        for data in tqdm(dataloader_test, total = len(dataloader_test), desc = "Mini-batch progress (Test)"):  
            cur_imgs = [x.split('/')[-1] for x in data['img_name']]
            img_list+=cur_imgs
            print("========================================",cur_imgs) 
            data_time.reset()
            decoder_labels_start = time.time()
            # The followig can be set using the distribution of the length
            max_source_length = 256

            T5encoding = T5tokenizer(
                data['T5_source1'],
                data['T5_source2'],
                padding="longest",
                max_length=max_source_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            T5input_ids, T5attention_mask = T5encoding.input_ids, T5encoding.attention_mask
            if code_prof:
                print(f"T5 input processing time: {data_time.val}")
            # ---------------------------------MODEL CODE---------------------------------
            model.zero_grad()
            data_time.reset()
            model_start = time.time() 
            # output_sequences = model.model_T5.generate(input_ids=T5input_ids,attention_mask=T5attention_mask,do_sample=False)
            output_sequences = model.generate(input_ids=T5input_ids,attention_mask=T5attention_mask,do_sample=False)
            generated_text = T5tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            generated_result+=generated_text
            if code_prof:
                print(f"model processing time: {data_time.val}")
            
            break
    return  generated_result, img_list, data['ent'], data['role'], data['caption']


# In[12]:


#exp_name = "E8_V2e_ViT_deBERTaLarge_T5_OCRENTROL_caption_15ep_3WTDlossMTL_BS4_Adafactor_LR1e-4_CorrectPrompt_maxsrctarlen512"
#exp_path = "saved/"+exp_name
path = os.path.join(config.MODEL_DIR, "final_t5model.pt")
criterion = nn.CrossEntropyLoss()
try:
    torch.cuda.empty_cache()
    gc.collect()
except:
    pass
# model = MM(n_out)
model = T5ForConditionalGeneration.from_pretrained("t5-large")
model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
model.to(device)

#get output
output = infer_model(model)
print("got output")
print("data captions generates",output)


# In[ ]:

"""
import anvil.server

anvil.server.connect('Enter Your Key Here')
# In[ ]:


import anvil.media
@anvil.server.callable
def generate_exp(file):
    with anvil.media.TempFile(file) as filename:
        cur_img = file.name
        # print(file)
        # img = load_img(filename)
        
        exp_name = "E8_V2e_ViT_deBERTaLarge_T5_OCRENTROL_caption_15ep_3WTDlossMTL_BS4_Adafactor_LR1e-4_CorrectPrompt_maxsrctarlen512"
        exp_path = "/home/shivams/memes/meme_exp_gen/saved/"+exp_name
        path = os.path.join(exp_path, "final_t5model.pt")
        criterion = nn.CrossEntropyLoss()
        try:
            del model
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        # model = MM(n_out)
        model = T5ForConditionalGeneration.from_pretrained("t5-large")
        model.load_state_dict(torch.load(path))
        model.to(device)
        # infer_model(model)

        
        generated_result, img_list, ent, role, caption = infer_model(model)
        for img_name, exp, e, r, c in zip(img_list, generated_result, ent, role, caption):
            if img_name == cur_img:
                print(img_name, cur_img)
                print(exp)
                return exp, e, r, c
"""

