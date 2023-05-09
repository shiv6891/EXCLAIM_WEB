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
        image =  test_data['image'].tolist()
        caption =  test_data['caption'].tolist()
        sentence = test_data['original'].tolist()
        ent = test_data['ent'].tolist()

        print(f"====>RoleLabel function: Sentence and Entity: {sentence}, {ent}",)
        model_results, probability_score = self.get_labels(test_data['sentence'].tolist(), test_data['ent'].tolist())
        #image = test_data['image'].tolist()

        # for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        #     sentence.append(row['original'])
        #     ent.append(row['ent'])
        #     # dataset_results.append(row['role'])
        #     label, score = self.get_labels(row['sentence'], row['ent'])
        #     model_results.append(label)
        #     probability_score.append(score)
        #     image.append(row['image'])
        df = pd.DataFrame()
        df['image_name']=image
        df['caption'] = caption
        df['sentence'] = sentence
        df['ent'] = ent
        df['model_results'] = model_results
        # df['dataset_results'] = dataset_results
        df['probability_score'] = probability_score
        #df['image'] = image
        df.to_csv(file_name, index=False)
        # print("Processing complete!")
        # print(model_results)
        return df
        # print(classification_report(dataset_results, model_results))
        

    def get_test_pdf_output(self, test_data, file_name):
        sentence = []
        ent = []
        model_results = []
        probability_score = []
        image = []
        for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
            sentence.append(row['original'])
            ent.append(row['ent'])
            label, score = self.get_labels(row['sentence'], row['ent'])
            model_results.append(label)
            probability_score.append(score)
            image.append(row['image'])
        df = pd.DataFrame()
        df['sentence'] = sentence
        df['ent'] = ent
        df['model_results'] = model_results
        df['probability_score'] = probability_score
        df['image'] = image
        df.to_csv(file_name, index=False)
        return df



    def get_labels(self, sentences, ents):
        # print(f"{'-'*40}\n Into the get_labels function now...")
        tokenized_sentence = self.tokenizer(sentences, ents, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')

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
        prob_out = torch.softmax(logits, dim=1).cpu().detach().numpy().tolist()
        final_output = np.argmax(prob_out, axis=1)

        # print(f"Final output processed successfully")
        # print(final_output.shape)
        # print(final_output)
        return [self.id2label[i] for i in final_output], prob_out
            
