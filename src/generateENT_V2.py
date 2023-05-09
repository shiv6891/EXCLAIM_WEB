import sys
sys.path.append('/home/azureuser/memes/memeExplain')
import config
import numpy as np
import os
if os.path.exists(config.DATA_PATH):
    data_path = config.DATA_PATH
else:
    data_path = '/home/azureuser/memes/memeExplain/Artifacts/Data'

# if os.path.exists(config.UPLOADS_PATH):
#     uploads_path = config.UPLOADS_PATH
# else:
#     uploads_path = '/home/azureuser/memes/memeExplain/static/uploads'


import spacy
import pandas as pd 
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
nlp = spacy.load('en_core_web_sm')
import json

try:
    import face_recognition
except ImportError as e:
    pass  # module doesn't exist, deal with it.
from tqdm.auto import tqdm
from collections import Counter
import pickle

# Using https://pypi.org/project/fuzzywuzzy/
# >>> from fuzzywuzzy import process
# >>> choices = ['Joe Biden', 'Barak Obama']
# >>> process.extractOne("jo bdn", choices)
# ('Joe Biden', 80)


# -------------------------------------------------------
# Initialize the reference data (standard entitites, visuals, etc.)
# -------------------------------------------------------
# with open(os.path.join(data_path, 'entset_DISHVV.json'), 'r') as ij:
# with open(os.path.join(data_path, 'entset_ExHVVTest_img2entlst.json'), 'r') as ij:
#     incon = json.load(ij)
# img2ent = incon
# print(len(img2ent.keys()))
# print(img2ent)


with open(os.path.join(data_path, 'known_ents_encodings.pkl'), 'rb') as ip:
    encdata = pickle.load(ip)
enc_list = encdata['known_ents_encodings']
known_ref_list = encdata['known_ents']


#df = pd.read_csv("hvvexp_test.csv")
#ocr_list = df["OCR"].tolist()
#ocr_set = set(ocr_list)
#ocr_set_list = list(ocr_set)


# -------------------------------------------------------
#               Extracting Textual Entities
# -------------------------------------------------------
def textENT(ocr, img):
    with open(os.path.join(data_path, 'entset_ExHVVAll_img2entlst.json'), 'r') as ij:
        incon = json.load(ij)
    img2ent = incon
    # final_text_ent = []
    #print("%%%%%%%%%%%%%%%%%",len(ocr))
    # for sentence in ocr:
    
    sentence = ocr

    finalenttemp = []
    doc = nlp(sentence)
    enttemp = []
    # print("====>NER stuff...")
    for ent in doc.ents:
        if ent.label_ in ['NORP','ORG','GPE','PERSON','LOC']:
            if len(ent.text) > 0:
                # print(f"====>\n{ent.text.lower(), ent.label_}")
                enttemp.append(ent.text.lower())
    # if len(enttemp) == 0:
    #     enttemp = ['Unknown_ENT']

    postemp = []
    # print("====>POS stuff...")
    for token in doc:
        # print(f"====>\n{token, token.pos_}")
        if token.pos_ in ["NOUN", "PROPN"]:
            postemp.append(token.text.lower())
    # if len(postemp) == 0:
    #     postemp = ['Unkown_POS']
    # print(enttemp, postemp)
    combinedsettemp = list(set(enttemp + postemp))

    # print("====>Combined stuff...")
    # print(f"====>combinedsettemp: {combinedsettemp}")
    
    # print("====>Standardizing steps...")
    # Standardizing the entitites via standard set (fuzzywuzzy!)
    if len(combinedsettemp)>0:
        for i in combinedsettemp:
            if len(i)==1:
                continue
            if i.lower().startswith('unkown_'):
                finalenttemp.append(i)
            else:
                try:
                    entout = process.extractOne(i.lower(), img2ent[img])
                    # print(f"====>\n{i, entout}")
                    if entout[1]>config.SIM_THRESH and np.abs(len(i)-len(entout[0]))<15:
                        if 'green' in entout[0] and 'green' not in sentence.lower():
                            continue
                        finalenttemp.append(entout[0])
                    else:
                        continue
                except:
                    finalenttemp.append(i)
    else:
        finalenttemp.append('UNKNOWN')

        
    # final_text_ent.append(finalenttemp)
    #print("%55555%%%%%%%%",len(entity_list)) 
    #print(entity_list)
    return finalenttemp
'''ocrproc = []
for row in df.iterrows():
    #sentence = "Apple is looking at buying U.K. startup for $1 billion"
    #s = "Trump 2 weeks ago: 'The Coronavirus is a Democrat Hoax' Trump today: I'm declaring Coronavirus a National Emergency."  
    ocr = row[1]["OCR"]
    #ocrproc.append(ocr)
    image = row[1]["image"]

    if ocr not in ocrproc:
        print("===================== OCR sentence is ==================> {}".format(ocr))
        doc = nlp(ocr) 
        for ent in doc.ents:
            #print("===================== OCR sentence is ==================> {}".format(ocr))
            #print("============")
            print("entity text ===>{}, label===>{}".format(ent.text,ent.label_))
            generatePOS(ent.text)
            #print(ent.text, ent.start_char, ent.end_char, ent.label_)
            #print("=============")
    ocrproc.append(ocr)
'''

# -------------------------------------------------------
#               Extracting Visual Entities
# -------------------------------------------------------
def visENT(path, unk_img_file):
    unknown_image = face_recognition.load_image_file(os.path.join(path, unk_img_file))
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    results = face_recognition.compare_faces(enc_list, unknown_encoding)
    odf = pd.DataFrame({'img': known_ref_list, 'bool': results})
    outimglist = odf[odf['bool']==True]['img'].tolist()
    procentlist = [' '.join(i.split('_')[:-1]) for i in outimglist]
    cntr = Counter(procentlist)
    srted_tuple = cntr.most_common()
    # print(f"\n====>Visual Entities stuff:\n{srted_tuple}")
    entcnt = len(srted_tuple)
    if entcnt:
        # cur_ent_list = [i[0] for i in srted_tuple]
        '''1. Even occuring once, some correct entity can 
        be placed 2nd, 3rd or higher in the tuple. 
        2. If srted_tuple[score][0] - srted_tuple[score][1]  > 0'''
        if entcnt>1 and (srted_tuple[0][1]-srted_tuple[1][1] > 1):
            cur_ent_list = [srted_tuple[0][0]]
        else:
            cur_ent_list = [i[0] for i in srted_tuple]
    else:
        cur_ent_list = []
    
    # print(f"====>V_NER returned list:\n{cur_ent_list}")
    return cur_ent_list

    
# -------------------------------------------------------
#       Main function for aggregating the entities
# -------------------------------------------------------
def generateENT(path, ocr, img):
    all_ents = []
    for cur_ocr, cur_img in tqdm(zip(ocr, img), desc="NER extraction:"):
        temp_ent = []
        temp_ent.extend(textENT(cur_ocr, cur_img))
        try:
            temp_ent.extend(visENT(path, cur_img))
        except:
            pass
        all_ents.append(list(set(temp_ent)))
        if len(all_ents[0]) == 0:
            all_ents.append("UNKNOWN")

        # '''Normalizing: Handle scenarios like: 
        # ['trump', 'donald trump', 'joe', 'joe biden']
        # ===>This can be commented out when not using the larger vocab'''
        # all_ent_norm = []
        # for cur_ent_set in all_ents:
        #     tmp_list = []
        #     for cur_ent in cur_ent_set:
        #         if 'trump' in cur_ent.lower() or 'donald' in cur_ent.lower():
        #             cur_ent = 'Donald Trump'
        #             tmp_list.append(cur_ent)
        #         elif 'obama' in cur_ent.lower() or 'barak' in cur_ent.lower():
        #             cur_ent = 'Barak Obama'
        #             tmp_list.append(cur_ent)
        #         elif 'biden' in cur_ent.lower() or 'joe' in cur_ent.lower():
        #             cur_ent = 'Joe Biden'
        #             tmp_list.append(cur_ent)
        #         else:
        #             tmp_list.append(cur_ent)
        #     all_ent_norm.append(list(set(tmp_list)))
    return all_ents


# def generatePOS(text):
#     #text = ("Trump 2 weeks ago: 'The Coronavirus is a Democrat Hoax' Trump today: I'm declaring Coronavirus a National Emergency.")
#     pos_list = []
#     doc = nlp(text)
#     print("==POS TAGS==")
#     # Token and Tag
#     for token in doc:
#         if token.pos_ == "PROPN":
#             pos_list.append(token.text.lower())
#     return pos_list
        

if __name__=="__main__":
    # sent_list = ["OBAMA: I want everything set up for Persident- Elect Trump's arrival, Joe. JOE: HEY LOOK WHAT | CAN DO OBAMA: Joe; your foot out of your mouth: JOE: get"]
    sent_list = ["Test sentence"]
    img_list = ["memes_6637.png"]
    finalentlist = generateENT(sent_list, img_list)
    print(finalentlist)

