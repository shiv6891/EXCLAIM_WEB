from src.utils.utilities import AverageMeter, data_dict
import torch
import truecase


# To be used for profiling the code (Checking the runtimes)
data_time = AverageMeter('Data', ':6.3f')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------------
#                   Explanation Generation       
# -------------------------------------------------------
def infer_model(model, T5tokenizer, df, img):
    model.eval()
    code_prof = False
    generated_result = []
    img_list = img
    # df = generateSample(path,img)
    data = data_dict(df)
    with torch.no_grad():
        #data_time.reset()
        #decoder_labels_start = time.time()
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
        model.zero_grad()
        #data_time.reset()
        #model_start = time.time()
        output_sequences = model.generate(input_ids=T5input_ids,attention_mask=T5attention_mask,do_sample=False)
        generated_text = T5tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        generated_result+=generated_text
        if code_prof:
            print(f"model processing time: {data_time.val}")
    
    generated_result_proc = [truecase.get_true_case(i) for i in generated_result]
    df['gen_explanation'] = generated_result_proc
    return df
            
    # return  {"GEN_RESULTS": generated_result,
    #         "IMG_LIST": img_list,
    #         "ENT_DATA": data['ent'],
    #         "ROLES": data['role'],
    #         "CAPTIONS": data['caption']}
# -------------------------------------------------------