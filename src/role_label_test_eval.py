from tqdm import tqdm
import torch
import config
# from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

MAX_LEN = 256

class EvalTest:
    def __init__(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cpu'
        #self.device = config.DEVICE
        self.id2label = {
            0 : 'other' ,
            1 : 'victim' ,
            2 : 'villain',
            3 : 'hero'
        } 

    def get_test_eval(self, test_data, file_name):
        sentence = []
        word = []
        model_results = []
        dataset_results = []
        probability_score = []
        #image = []
        #print(test_data)
        for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
            sentence.append(row['original'])
            word.append(row['word'])
            #print("senteces and words are printed here :====",row["sentence"],row["word"])
            #dataset_results.append(row['role'])
            label, score = self.get_labels(row['sentence'], row['word'])
            #print("score and labels are printed here:======",label,score)
            model_results.append(label)
            probability_score.append(score)
            #image.append(row['image'])
        df = pd.DataFrame()
        df['sentence'] = sentence
        df['word'] = word
        df['model_results'] = model_results
        #df['dataset_results'] = dataset_results
        df['probability_score'] = probability_score
        #df['image'] = image
        df.to_csv(file_name, index=False)
        #print("Processing complete!")
        print("====================",model_results)
        return model_results[0]
        # print(classification_report(dataset_results, model_results))
        

    def get_test_pdf_output(self, test_data, file_name):
        sentence = []
        word = []
        model_results = []
        probability_score = []
        image = []
        for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
            sentence.append(row['original'])
            word.append(row['word'])
            label, score = self.get_labels(row['sentence'], row['word'])
            model_results.append(label)
            probability_score.append(score)
            image.append(row['image'])
        df = pd.DataFrame()
        df['sentence'] = sentence
        df['word'] = word
        df['model_results'] = model_results
        df['probability_score'] = probability_score
        df['image'] = image
        df.to_csv(file_name, index=False)
        return df



    def get_labels(self, sentence, word):
        tokenized_sentence = self.tokenizer(sentence, word, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(
                tokenized_sentence['input_ids'].to(self.device), 
                tokenized_sentence['attention_mask'].to(self.device),
                # tokenized_sentence['token_type_ids'].to(self.device)
                )
            # outputs = self.model(
            #     tokenized_sentence['input_ids'].cuda(), 
            #     tokenized_sentence['attention_mask'].cuda(),
            #     # tokenized_sentence['token_type_ids'].cuda()
            #     )

        logits = outputs.logits
        final_output = np.argmax(torch.softmax(logits, dim=1).cpu().detach().numpy().tolist(), axis=1)

        return self.id2label[final_output[0]], torch.softmax(logits, dim=1).cpu().detach().numpy().tolist()
            
