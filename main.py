# -------------------------------------------------------
# main.py: Main Flask app script
'''Main code for running the Flask app 
for Meme Explanation Generation Application'''
# -------------------------------------------------------


# -------------------------------------------------------
# Library imports
# -------------------------------------------------------
import random
import config
from tqdm.auto import tqdm
from tqdm import trange 
import os
import time
import warnings
import numpy as np
import json
from prettytable import PrettyTable
warnings.filterwarnings("ignore")



from flask import Flask, request, render_template

from src.caption_module import caption
from src.generateENT import generateENT as rawEntEx
from src.generateENT_V2 import generateENT as stdEntEx
from src.ocr import gocr as tessocr
from src.ocr_V2 import gocr as easyocr
from src.role_label_inference import generateRole
from src.gen_exp import infer_model
from src.utils.utilities import create_directory, uploadNsave, get_final_dict

import torch
from transformers import OFATokenizer, OFAModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.utils import logging
from PIL import Image
# -------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING=1
#torch.cuda.set_device(1)

# logging.set_verbosity(40)

# -------------------------------------------------------
#                   Loading Models
# -------------------------------------------------------

model_dir = config.MODEL_DIR
t5_model_name = config.T5_variant
t5_model_path = os.path.join(model_dir, t5_model_name)
out_directory_ch = [config.DATA_PATH, config.CACHE_PATH, config.MODEL_DIR, t5_model_path]
create_directory(out_directory_ch)

dir_list = os.listdir(t5_model_path)
if dir_list:
    T5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
    T5tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
else:
    T5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name, cache_dir=config.CACHE_PATH)
    T5tokenizer = T5Tokenizer.from_pretrained(t5_model_name, cache_dir=config.CACHE_PATH)
    T5_model.save_pretrained(t5_model_path)
    T5tokenizer.save_pretrained(t5_model_path)

T5_LUMEN_path = os.path.join(model_dir, "final_t5model.pt")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
T5_model.load_state_dict(torch.load(T5_LUMEN_path,map_location=torch.device('cpu')))
T5_model.to(device)

ofa_model_path = config.OFA_PATH
ofa_tokenizer = OFATokenizer.from_pretrained(ofa_model_path)
ofa_model = OFAModel.from_pretrained(ofa_model_path, use_cache=False)


# -------------------------------------------------------
#           Calling all the SUBMODULES involved
# -------------------------------------------------------

def getMetaData(path,img):
    '''Calls each submodules
    Returns: A dataframe of resuts and metadata'''

    # 1. Caption Generation
    caption_out = caption(ofa_tokenizer, ofa_model, path, img)
    print('-'*100)
    print(f"====>Check2: Caption results: {caption_out}")
    # 2. OCR generation
    # ocr = tessocr(path,img)
    ocr_alph_l, ocr_preproc = easyocr(path,img)
    print('-'*100)
    print(f"====>Check3: OCR results: {ocr_alph_l}")
    # 3. Entity Generation yet to be Implement
    # ent = rawEntEx(oce_preproc)
    print(f"====>Check 3-4: {path}, {ocr_preproc}, {img}")
    ent = stdEntEx(path, ocr_preproc, img)    
    print('-'*100)
    print(f"====>Check4: Entity results: ")
    print(ent)
    #ent = 'Donal Trump'
    # 4. Role Generation
    roledf = generateRole(img,caption_out,ocr_alph_l,ent)
    print('-'*100)
    print(f"====>Check5: Role-label results")
    print(roledf.head())
    t = PrettyTable(['Entity', 'Role', 'Probability', 'Caption', 'OCR'])
    for row in roledf.iterrows():
        data = row[1]
        t.add_row([data.ent, data.model_results, np.round(np.max(data.probability_score), 2), data.caption,
                   data.sentence])
    print(t)
    '''Any potential filtering of the dataframe (roledf) 
    can be done at this stage before returning.'''
    return roledf


# -------------------------------------------------------
#    Initiate the process and return the final paylod
# -------------------------------------------------------
def process_initiate(start_time, common_img_path, img_list):
    '''Get the meta data, that's pre-requisite
    for explanation generation'''
    image_list = list(set(img_list))
    print('-'*100)
    print(f"====>Check1: Image file list: {image_list}")
    samplesdf = getMetaData(common_img_path, image_list)

    # Final LUMEN-based inferencing
    t5_out_df = infer_model(T5_model, T5tokenizer, samplesdf, image_list)
    print('-'*100)
    print("====>Check6: T5 Output Details")    
    image_list = t5_out_df['image_name'].tolist()
    ocr_list = t5_out_df['sentence'].tolist()
    cap_list = t5_out_df['caption'].tolist()
    genexp_list = t5_out_df['gen_explanation'].tolist()
    
    # print('-'*100)
    # print(f"====>Check6a: Image file list: {image_list}")
    # print('-'*100)
    # print(f"====>Check6b: OCR list: {ocr_list}") 
    # print('-'*100)
    # print(f"====>Check6c: Caption list: {cap_list}")
    # print('-'*100) 
    # print(f"====>Check6d: Generated Explanation list: {genexp_list}") 

    # Final format conversion
    final_dict = get_final_dict(t5_out_df)
    print('-'*100)
    print(f"Final dictionary obatined")
    print(json.dumps(final_dict, indent=3))
    with open(os.path.join(config.DATA_PATH, 'final_out.json'), 'w') as jo:
        jo.write(json.dumps(final_dict, indent=3))    
    timeelapsed = np.round(time.time() - start_time, 2)
    print('-'*100)
    print("====>Time Elapsed %s seconds ---" % (timeelapsed))
    content = {'timeel': timeelapsed, 'N': len(set(image_list)), 'img_list': image_list, 'exp_list': genexp_list, 'json': final_dict}
    return content


# -------------------------------------------------------
#                   FLASK API CALLING              
# -------------------------------------------------------
app = Flask(__name__)

upload_folder = 'static/uploads'
create_directory([upload_folder])

app.config['UPLOAD_FOLDER'] = upload_folder

# @app.route('/')
# def index():
#     f2rem = os.listdir(config.UPLOADS_PATH)
#     if len(f2rem)>100:
#         for f in f2rem:
#             os.remove(os.path.join(config.UPLOADS_PATH, f))
#     return render_template('index.html')

@app.route('/')
def index():
    f2rem = os.listdir(config.UPLOADS_PATH)
    if len(f2rem)>100:
        for f in f2rem:
            os.remove(os.path.join(config.UPLOADS_PATH, f))
    samples_ent1 = os.listdir(config.SAMPLE_PATH + '/1ent/')
    samples_ent2 = os.listdir(config.SAMPLE_PATH + '/2ent/')
    samples_ent3 = os.listdir(config.SAMPLE_PATH + '/3ent/')
    samples_ent4 = os.listdir(config.SAMPLE_PATH + '/4ent/')
    se1, se2, se3, se4 = [], [], [], []
    for item in samples_ent1:
        se1.append(os.path.join(config.SAMPLE_PATH + '/1ent/', item))
    for item in samples_ent2:
        se2.append(os.path.join(config.SAMPLE_PATH + '/2ent/', item))
    for item in samples_ent3:
        se3.append(os.path.join(config.SAMPLE_PATH + '/3ent/', item))
    for item in samples_ent4:
        se4.append(os.path.join(config.SAMPLE_PATH + '/4ent/', item))

    data_content = {"ent1": se1, "ent2": se2, "ent3": se3, "ent4": se4}

    return render_template('index.html', data=data_content)


# demo
@app.route('/demo')
def demo():
    f2rem = os.listdir(config.UPLOADS_PATH)
    if len(f2rem)>100:
        for f in f2rem:
            os.remove(os.path.join(config.UPLOADS_PATH, f))
    samples_ent1 = os.listdir(config.SAMPLE_PATH + '/1ent/')
    samples_ent2 = os.listdir(config.SAMPLE_PATH + '/2ent/')
    samples_ent3 = os.listdir(config.SAMPLE_PATH + '/3ent/')
    samples_ent4 = os.listdir(config.SAMPLE_PATH + '/4ent/')
    se1, se2, se3, se4 = [], [], [], []
    for item in samples_ent1:
        se1.append(os.path.join(config.SAMPLE_PATH + '/1ent/', item))
    for item in samples_ent2:
        se2.append(os.path.join(config.SAMPLE_PATH + '/2ent/', item))
    for item in samples_ent3:
        se3.append(os.path.join(config.SAMPLE_PATH + '/3ent/', item))
    for item in samples_ent4:
        se4.append(os.path.join(config.SAMPLE_PATH + '/4ent/', item))

    data_content = {"ent1": se1, "ent2": se2, "ent3": se3, "ent4": se4}

    return render_template('demo.html', data=data_content)


@app.route('/processSelection', methods=['POST'])
def processSelection():
    start_time = time.time()
    print("====>Inside DEMO Mode")
    print(request.values)
    images_selected = request.form['cartholder']
    print(f"====>Step1: images_selected:\n{images_selected}")
    print(f"Type identified (expected: str): {type(images_selected)}")
    print(f"Length of the returned content: {len(images_selected)}") 
    if len(images_selected)==0:
        return render_template('error.html') 
    else:
        potential_list = json.loads(images_selected)
        img_paths = [json.loads(i)['productName'] for i in potential_list]
        print(f"====>Step2: img_paths:\n{img_paths}")
        image_list = [i.split('/')[-1] for i in img_paths]
        print(f"====>Step3: image_list:\n{image_list}")
        content = process_initiate(start_time, config.SAMPLE_PATH, image_list)    
        return render_template('dummy_display.html', data=content)

@app.route('/upload', methods=['POST'])
def upload():
    start_time = time.time()
    # Upload files and collect details 
    image_list = []
    images = request.files.getlist('image')
    image_list = uploadNsave(images, upload_folder)    
    content = process_initiate(start_time, upload_folder, image_list)    
    return render_template('dummy_display.html', data=content)

if __name__ == '__main__':
    # port = 5000 + random.randint(0,999)
    # print(port)
    app.config["DEBUG"] = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(port=5002, host = "0.0.0.0")

