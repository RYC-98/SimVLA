import argparse
import os
# import ruamel.yaml as yaml
from ruamel.yaml import YAML 
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from transformers import BertForMaskedLM

from models.model_ve import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import ve_dataset
from PIL import Image
from torchvision import transforms
from attack import *
from models import clip

from SimVLA import SimAttacker, ImageAttacker, TextAttacker      

def load_model(model_name, model_ckpt, text_encoder, device):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)    
    if model_name in ['ALBEF', 'TCL']:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)

        checkpoint = torch.load(model_ckpt, map_location='cpu')
    ### load checkpoint
    else:
        model, preprocess = clip.load(model_name, device=device)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer
    
    try:
        state_dict = checkpoint['model'] # tcl
    except:
        state_dict = checkpoint          # albef

    if model_name == 'TCL':
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        # 
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]  
            del state_dict[key]                     
    
    model.load_state_dict(state_dict, strict=False) 
    
    return model, ref_model, tokenizer

def evaluate(model, ref_model, t_model, t_ref_model, data_loader, tokenizer, device, config, t_test_transform):
    model.float()      
    model.eval()
    ref_model.eval()
    t_model.float()
    t_model.eval()
    t_ref_model.eval()     

    if args.scales is not None:
        scales = [float(itm) for itm in args.scales.split(',')]
        print(scales)
    else:
        scales = None

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    img_attacker = ImageAttacker(images_normalize, eps=2/255, steps=10, step_size=0.5/255)
    txt_attacker = TextAttacker(ref_model, tokenizer, cls=False, max_length=30, number_perturbation=1,
                                topk=10, threshold_pred_score=0.3)
    attacker = SimAttacker(model, img_attacker, txt_attacker)



    for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        images, targets = images.to(device), targets.to(device)

        txt2img = []
        for i in range(len(images)):
            txt2img += [i]
            
        images, text = attacker.attack(images, text, txt2img, device=device,
                                                    max_lemgth=30) 
        t_adv_img_list = []
        for itm in images:
            t_adv_img_list.append(t_test_transform(itm))
        images = torch.stack(t_adv_img_list, 0).to(device)  

        
        images = images_normalize(images)
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        with torch.no_grad():
            prediction = t_model(images, text_inputs, targets=targets, train=False)

        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0) 

        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    device = args.gpu[0]

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


    #### Model ####
    print("Creating model")
    print('source:', args.source_model, 'target:', args.target_model)
    print('source_ckpt:', args.source_ckpt, 'target_ckpt:', args.target_ckpt)

    model, ref_model, tokenizer = load_model(args.source_model, args.source_ckpt, args.source_text_encoder, device)
    t_model, t_ref_model, t_tokenizer = load_model(args.target_model, args.target_ckpt, args.target_text_encoder, device)
 
    model = model.to(device)
    ref_model = ref_model.to(device) 

    t_model = t_model.to(device)
    t_ref_model = t_ref_model.to(device)


    # model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    # ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)

    # checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # try:
    #     state_dict = checkpoint['model']
    # except:
    #     state_dict = checkpoint

    # # reshape positional embedding to accomodate for image resolution change
    # pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    # state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped


    # # msg = model.load_state_dict(state_dict, strict=False)
    # print('load checkpoint from %s' % args.checkpoint)
    # #print(msg)

    # model = model.to(device)
    # ref_model = ref_model.to(device)


    #### Dataset ####
    print("Creating dataset")
    # test_transform = transforms.Compose([
    #     transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
    #     transforms.ToTensor(),
    # ])

    n_px = model.visual.input_resolution
    s_test_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),       
    ])

    t_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),        
    ])

    datasets = ve_dataset(config['test_file'], s_test_transform, config['image_root'])
    test_loader = DataLoader(datasets, batch_size=config['batch_size_test'], num_workers=4)

    tokenizer = BertTokenizer.from_pretrained(args.source_text_encoder)



    print("Start evaluating")
    start_time = time.time()

    test_stats = evaluate(model, ref_model, t_model, t_ref_model, test_loader, tokenizer, device, config, t_test_transform)

    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'eval type': args.adv, 'cls': args.cls,
                 'eps': config['epsilon'], 'iters':config['num_iters'], 'alpha': args.alpha}

    # with open(os.path.join(args.output_dir, "log.txt"), "a+") as f:
    #     f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VE.yaml')
    parser.add_argument('--output_dir', default='output/VE')


    parser.add_argument('--source_model', default='RN101', type=str)
    # parser.add_argument('--source_text_encoder', default='bert-base-uncased', type=str)  # source
    parser.add_argument('--source_text_encoder', default='./checkpoints/bert-base-uncased', type=str) 
    parser.add_argument('--source_ckpt', default='', type=str)      

    parser.add_argument('--target_model', default='ALBEF', type=str)
    # parser.add_argument('--target_text_encoder', default='bert-base-uncased', type=str)  # source
    parser.add_argument('--target_text_encoder', default='./checkpoints/bert-base-uncased', type=str)
    parser.add_argument('--target_ckpt', default='./checkpoints/albef_ve.pth', type=str)   


    parser.add_argument('--gpu', type=int, nargs='+',  default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--adv', default=4, type=int)
    # parser.add_argument('--cls', action='store_true')
    parser.add_argument('--cls', action='store_true') # the output of CLIP is [CLS] embedding, so needn't to select at 0 
    parser.add_argument('--alpha', default=3.0, type=float)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
    args = parser.parse_args()

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    yaml = YAML(typ='rt')                      
    config = yaml.load(open(args.config, 'r', encoding='utf-8')) 

    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
