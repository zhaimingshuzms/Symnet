import json
import os
import sys

DEFAULT_ROOT = "aux_data"

class TextLoader:
    def __init__(self, attr_file_path, obj_file_path):
        self.attr_file_path = attr_file_path
        self.obj_file_path = obj_file_path

    def get_attr_dict(self):
        path = os.path.join(os.path.dirname(sys.argv[0]),DEFAULT_ROOT,self.attr_file_path)
        with open(path,"r") as f:
            dict = json.load(f)
        f.close()
        return dict
    
    def get_obj_dict(self):
        path = os.path.join(os.path.dirname(sys.argv[0]),DEFAULT_ROOT,self.obj_file_path)
        with open(path,"r") as f:
            dict = json.load(f)
        f.close()
        return dict

    def get_type(self):
        return self.attr_file_path.split('_')[0]

if __name__=="__main__":
    textloader = TextLoader("UT_attrs.json","UT_objs.json")
    print(textloader.get_obj_dict())