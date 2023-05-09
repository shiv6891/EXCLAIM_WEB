import spacy
import pandas as pd 
nlp = spacy.load('en_core_web_sm')


#df = pd.read_csv("hvvexp_test.csv")

#ocr_list = df["OCR"].tolist()

#ocr_set = set(ocr_list)

#ocr_set_list = list(ocr_set)


def generateENT(ocr):

    entity_list = []
    #print("%%%%%%%%%%%%%%%%%",len(ocr))
    for sentence in ocr:
        doc = nlp(sentence)
        temp = []
        for ent in doc.ents:
            if ent.label_ in ['NORP','ORG','GPE','PERSON','LOC']:
                if len(ent.text) > 0:
                    temp.append(ent.text)
        if len(temp) > 0:
            entity_list.append(temp)
        else:
            entity_list.append(['Unknown'])
    #print("%55555%%%%%%%%",len(entity_list)) 
    #print(entity_list)
    return entity_list
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
def generatePOS(text):
    #text = ("Trump 2 weeks ago: 'The Coronavirus is a Democrat Hoax' Trump today: I'm declaring Coronavirus a National Emergency.")
  
    doc = nlp(text)
    print("==POS TAGS==")
    # Token and Tag
    for token in doc:
        
        print(token, token.pos_)
        
    # You want list of Verb tokens
    print("Verbs:", [token.text for token in doc if token.pos_ == "VERB"])

# generateENT(['Apple is looking at buying U.K. startup for $1 billion',"Trump 2 weeks ago: 'The Coronavirus is a Democrat Hoax' Trump today:"])
#generatePOS()
