try:
    import config
except:
    import sys
    sys.path.append('/home/azureuser/memes/memeExplain')
    import config
import json
import os
import logging as sys_log
import numpy as np
import regex as re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import pandas as pd
try:
    import face_recognition
except ImportError as e:
    pass  # module doesn't exist, deal with it.
from tqdm.auto import tqdm
import truecase
import pickle


# -------------------------------------------------------
#                   Code Profiling Class
# -------------------------------------------------------
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


# -------------------------------------------------------
#                   Formatting Data       
# -------------------------------------------------------
def  data_dict(data):
    T5_source2 = data['caption'].tolist()
    T5_source1 = data.apply(lambda x: "Generate explanation for "+x['ent']+" as "+x['model_results']+": "+x['sentence'].replace(' .', '.'), axis=1)

    #T5_source1 = "Generation explanation for "+ent+" as "+role+": "+ocr.replace('\n', ' ').replace(' .', '.')
    #T5_source2 = caption

    sample = {
         "img_name": data['image_name'].tolist(),
         "T5_source1": T5_source1.tolist(),
         "T5_source2": T5_source2,
         "ent": data['ent'].tolist(),
         "role": data['model_results'].tolist(),
         "caption": data['caption'].tolist()
        }
    return sample

# -------------------------------------------------------
#                   Creating directories
# -------------------------------------------------------
def create_directory(dirs: list) -> None:
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        sys_log.info(f"directory is created at {dir_path}")


# -------------------------------------------------------
#                   Saving uploaded files
# -------------------------------------------------------
def uploadNsave(images, upload_folder):
    image_list = []
    for file in images:
        file.save(os.path.join(upload_folder, file.filename))
        image_list.append(file.filename)
    return image_list


# -------------------------------------------------------
#                   Preparing the Final Dictionary
# -------------------------------------------------------
def get_final_dict(indf):
    uniq_img_list = list(set(indf.image_name.tolist()))
    final_out_dict = {}
    print("====>Preparing the final dict")
    for img in uniq_img_list:
        final_out_dict[img] = []
        curdf = indf[indf['image_name']==img]
        for row in curdf.iterrows():
            print(truecase.get_true_case(row[1].caption),truecase.get_true_case(row[1].sentence))
            final_out_dict[img].append({'caption':truecase.get_true_case(row[1].caption),
                                        'ocr':truecase.get_true_case(row[1].sentence),
                                        'ent': truecase.get_true_case(row[1].ent),
                                        'role': row[1].model_results,
                                        'prob': np.round(np.max(row[1].probability_score), 2),
                                        'exp': row[1].gen_explanation})
    return final_out_dict


# -------------------------------------------------------
#  Code for offline encoding of known images (VIS. NER)
# -------------------------------------------------------
def offlineencode():
    # -------------------------------------------------------
    '''Entity Recognition (Face Recognition)
    This code processes the pre-stored "known" image references.
    These can be encoded using "face_recognition" API.
    Can be later compared with an unknown image encoding.
    API URL: https://github.com/ageitgey/face_recognition'''
    # -------------------------------------------------------
    root_path = '/home/azureuser/memes/testimages' # Parent directory for the image folder to be processed
    ref_path = os.path.join(root_path, 'Ref_Images') # Should contain reference image to be encoded
    known_image_list = os.listdir(ref_path)
    # print(known_image_list)
    known_ents = []
    known_ents_encodings = []
    for im in tqdm(known_image_list):
        try:
            known_image = face_recognition.load_image_file(os.path.join(ref_path, im))
            known_encoding = face_recognition.face_encodings(known_image)[0]        
        except:
            continue
        known_ents.append(im)
        known_ents_encodings.append(known_encoding)
    with open(os.path.join(config.DATA_PATH, 'known_ents_encodings.pkl'), 'wb') as op:
        pickle.dump({'known_ents': known_ents, 'known_ents_encodings': known_ents_encodings}, op)

# -------------------------------------------------------


# -------------------------------------------------------
#          Creating img to entity list dictionary
# -------------------------------------------------------
def img_entlistmapper():
    indf = pd.read_csv('/home/azureuser/memes/memeExplain/Artifacts/Data/hvvexp_val.csv', index_col=0)
    print(indf.columns)
    # return indf
    # Check and initialize a pre-existing dictionary if allready there
    # entset_ExHVVTest_img2entlst.json
    try:
        with open(os.path.join('/home/azureuser/memes/memeExplain/Artifacts/Data', 'entset_ExHVVAll_img2entlst.json'), 'r') as ji:
            img2ent_map = json.load(ji)
        print("Found an existing file to initialize from")
    except:
        img2ent_map = dict()
    img_list = [a.strip() for a in list(set(indf['image'].tolist()))]
    for cur_img in tqdm(img_list):
        cur_entlist = list(set(indf[indf['image']==cur_img]['entity'].tolist()))
        if cur_img not in img2ent_map:
            img2ent_map[cur_img] = cur_entlist
        else:
            img2ent_map[cur_img] = img2ent_map[cur_img].extend(cur_entlist)
    print('-'*50)
    print('Image 2 Entity Mapping Complete')
    print(f"Original image set count: {len(img_list)}")
    print(f"Ditionary keys count (after populating): {len(img2ent_map.keys())}")
    # assert len(img2ent_map.keys())==len(img_list), "Lenth Mismatch! Check again."
    with open(os.path.join('/home/azureuser/memes/memeExplain/Artifacts/Data', 'entset_ExHVVAll_img2entlst.json'), 'w') as jo:
        json.dump(img2ent_map, jo)


# -------------------------------------------------------
#          Preprocessig the input OCR
# -------------------------------------------------------
def preproctext(inocr):
    alphregx = '[^A-Za-z0-9 ]+'
    ocr_alph_l = []
    ocr_without_sw_l = []
    for ocr in inocr:
        ocr_alph = re.sub(alphregx, '', ocr)
        # text_tokens = word_tokenize(ocr_alph)
        # ocr_without_sw = ' '.join([word for word in text_tokens if not word in stopwords.words()])
        ocr_alph_l.append(ocr_alph)
        ocr_without_sw_l.append(ocr_alph)
    return ocr_alph_l, ocr_without_sw_l

if __name__=='__main__':
    img_entlistmapper()