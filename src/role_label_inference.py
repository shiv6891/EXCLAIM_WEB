import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import os
import pandas as pd
from src.role_label_test_eval_batch import EvalTest

import transformers
# import argparse
import re
transformers.logging.set_verbosity_error()

# parser = argparse.ArgumentParser()

# parser.add_argument('-m', '--model', nargs='*', default='deberta_large', help='The name of the model to be evaluated.')

# args = parser.parse_args()
# model = args.model[0]

model = config.MODEL_DEBERTA

# print(model)

def process_data(img,cap,ocr,ent_list):
    
    # Create and return a df for a total of N samples
    N = 5 # can be dynamically set as the total samples to be processed
    original = [] # Raw OCR from meme image
    text = [] # processed OCR text
    ent = [] # entity label being probed
    image = [] # image name
    caption = [] # captions of the images

    possible_OCRs = [
        "Donald Trump has been completely responsible for all the human index development in America.",
        "Donald Trump is completely responsible for all the human rights violation that America has ever seen.",
        "Donald Trump has been completely ignored during his presidency.",
        "Donald Trump was the previous president of United States of America.",
        "Is there anything that Donald Trump hasn't spoiled yet?"

    ]
 
    for i in range(len(ocr)):
        # print(i)
        #original_ocr = possible_OCRs[i]
        original_ocr = ocr[i]
        original.append(original_ocr) 
        
        text.append(original_ocr.lower().replace('\n', ' '))
        ent.append(ent_list[i])
        image.append(img[i])
        caption.append(cap[i])
        #image.append("text_meme_{}.jpg".format(str(i)))
    
    #print("@@@@@@@@@",ocr,ent)
    df_test = pd.DataFrame()
    df_test['image'] = image
    df_test['caption']=caption
    df_test['sentence'] = text #[ocr.lower().replace('\n', ' ')]
    df_test['original'] = original #[ocr]
    df_test['ent'] = ent #[ent]
    #df_test['image'] = [img]
    return df_test

# print('------------------------------------------')
# print(model)
if model == 'deberta_large':
    MODEL_NAME = "microsoft/deberta-v3-large"
    MODEL_STORING_PATH = os.path.join(config.MODEL_DIR, 'best_model_deberta_large')
else:
    MODEL_NAME = "microsoft/deberta-v3-small"
    MODEL_STORING_PATH = os.path.join(config.MODEL_DIR, 'best_model_deberta_small')


# -------------------------------------------------------------------------------------
# Define processing module (replaces the module that returns the test set dataframe)
# This can be reverse mapped to the pipeline feeding meme details: oce, entity, image name etc.

def generateRole(image,caption,ocr,ent):
    image_list = []
    caption_list = []
    ent_list = []
    ocr_list = []
    for i in range(len(ocr)):
        for j in range(len(ent[i])):
            image_list.append(image[i])
            caption_list.append(caption[i])
            ocr_list.append(ocr[i])
            ent_list.append(ent[i][j])
    #print("===============",ocr_list,ent_list)

    data_test = process_data(image_list,caption_list,ocr_list,ent_list)

    # data_test = get_gold_label_test()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    #device = torch.device(config.DEVICE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    model.load_state_dict(torch.load(MODEL_STORING_PATH, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    evals = EvalTest(tokenizer, model)
    result = evals.get_test_eval(data_test, os.path.join(config.DATA_PATH, f"test_output_file_{re.sub('[^a-zA-Z0-9]', '_',MODEL_NAME)}.csv"))
    return result
