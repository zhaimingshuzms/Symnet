import clip
import numpy as np
import torch
import os
from TextLoader import TextLoader
DEFAULT_ROOT = "aux_data"

def load_clip():
    print(clip.available_models())
    model, preprocess = clip.load("ViT-L/14")
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    return model,preprocess

def write_into(path, content):
    if os.path.isfile(path):
        raise FileExistsError("previously generated file exists")
    
    with open(path,"w") as f:
        f.write(content)
    f.close()

def tostr(name,dict):
    return name + " = " + str(dict) + "\n"

def run():
    model , preprocess = load_clip()
    Loader = TextLoader("UT_attrs.json","UT_objs.json")
    
    keys = Loader.get_attr_dict().keys()
    texts = Loader.get_attr_dict().values()
    tokens = clip.tokenize(texts).cuda()
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    attrs_dict = dict(zip(keys,text_features.tolist()))

    keys = Loader.get_obj_dict().keys() 
    texts = Loader.get_obj_dict().values()
    tokens = clip.tokenize(texts).cuda()
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    objs_dict = dict(zip(keys,text_features.tolist()))

    filename = "my_"+Loader.get_type()+".py"
    path = os.path.join(DEFAULT_ROOT,filename)
    content = tostr("attrs_dict",attrs_dict) + tostr("objs_dict",objs_dict)
    write_into(path, content)

if __name__=='__main__':
    run()