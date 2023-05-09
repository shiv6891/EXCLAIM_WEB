from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator
import torch
import os

def generateCaption(tokenizer, model, img):
    #img = Image.open(img)
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 256

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    # ckpt_dir='./OFA-base'
    #ckpt_dir='./OFA-large'
    # ckpt_dir = '/home/azureuser/memes/OFA-base'
    #ckpt_dir='./OFA-huge'
    #ckpt_dir='./OFA-tiny'
    # tokenizer = OFATokenizer.from_pretrained(ckpt_dir)


    txt = " what does the image describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids

    #img = Image.open(image)
    patch_img = patch_resize_transform(img).unsqueeze(0)

    # model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)

    """## Choice of Generators
    We find that using our provided generator can consistently achieve a better performance on the benchmark evaluation. Therefore, we first provide a demonstration of how to use this generator, and later the native one from Transformers.
    """

    generator = sequence_generator.SequenceGenerator(
        tokenizer=tokenizer,
        beam_size=5,
        max_len_b=16,
        min_len=0,
        no_repeat_ngram_size=3,
    )

    data = {}
    data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}

    gen_output = generator.generate([model], data)
    gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

    #display(img)
    caption = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
    #print("caption for the image is :  ",tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip())
    return caption

def caption(ofa_tokenizer, ofa_model, path,image_list):
    g_caption = []
    for img in image_list:
        img = Image.open(os.path.join(path,img))
        cap = generateCaption(ofa_tokenizer, ofa_model, img)
        g_caption.append(cap)
    return g_caption
