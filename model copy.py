from tqdm.notebook import tqdm
from collections import defaultdict


def map_to_caption(caption_path):
    with open(caption_path, "rb") as f:
        next(f)
    captions_doc = f.read().decode('utf-8')

    mapping = defaultdict(list)
    for line in tqdm(captions_doc.split('\n')):
        #split line by ,
        tokens = line.split(",")
        if len(tokens)<2:
            continue
        image_id, caption = tokens[0], " ".join(tokens[1:])
        image_id = image_id.split(".")[0]
        
        mapping[image_id].append(caption)
    return mapping

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            #One caption at a time
            caption = captions[i]
            
            #preprocess
            caption = caption.lower()
            caption = caption.replace("[^A-Za-z0-9]", "") #remove numbers too
            caption = caption.replace("\s+", " ")
            
            #start and end tags to the caption
            caption = "startseq "  + " ".join(word for word in caption.split() if len(word)>1) + " endseq" #better to set len == 1
            captions[i] = caption