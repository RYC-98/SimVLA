import json 
import os
import re

import torch 
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])", '', caption.lower(),).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')


    caption = re.sub(
        r"\s{2,}",                      
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')      
    caption = caption.strip(' ')        

    caption_words = caption.split(' ')  
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])  
            
    return caption


class paired_dataset(Dataset):
    '''
    paired_dataset(config['test_file'], s_test_transform, config['image_root'])
    config['test_file'] = './data_annotation/flickr30k_test.json'
    config['image_root'] = './data/'
    '''
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r')) # './data_annotation/flickr30k_test.json'
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []

        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for i, ann in enumerate(self.ann): 
            self.img2txt[i] = []
            self.image.append(ann['image'])       
            for j, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words)) 
                self.txt2img[txt_id] = i          
                self.img2txt[i].append(txt_id)    
                txt_id += 1                       

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)   
        text_ids =  self.img2txt[index] 
        texts = [self.text[i] for i in self.img2txt[index]]
        return image, texts, index, text_ids
# self.text = [
#     "A cat is sitting on the mat.",       # text_id 0
#     "The cat looks very content.",        # text_id 1
#     "A small cat with green eyes.",       # text_id 2
#     "The mat is colorful.",               # text_id 3
#     "The cat is purring.",                # text_id 4
#     "A dog is running in the park.",      # text_id 5
#     "The dog is chasing a ball.",         # text_id 6
#     "A brown dog with floppy ears.",      # text_id 7
#     "The park is full of trees.",         # text_id 8
#     "The dog seems very happy."           # text_id 9
# ]


    def collate_fn(self, batch):

        imgs, txt_groups, img_ids, text_ids_groups = list(zip(*batch))      

        # (imgs1, txt_groups1, list(img_ids)1, text_ids_groups1)，(imgs2, txt_groups2, list(img_ids)2, text_ids_groups2)

        imgs = torch.stack(imgs, 0) 
        return imgs, txt_groups, list(img_ids), text_ids_groups

# Example:
# >>> imgs.shape
# torch.Size([2, 3, 384, 384])
# >>> texts_group
#(['the man with pierced ears is wearing glasses and an orange hat', 'a man with glasses is wearing a beer can crocheted hat', 'a man with gauges and glasses is wearing a blitz hat', 'a man in an orange hat starring at something', 'a man wears an orange hat and glasses'], 
# ['a black and white dog is running in a grassy garden surrounded by a white fence', 'a boston terrier is running on lush green grass in front of a white fence', 'a black and white dog is running through the grass', 'a dog runs on the green grass near a wooden fence', 'a boston terrier is running in the grass'])
# # >>> images_ids
# [0, 1]
# >>> text_ids_groups
# ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])