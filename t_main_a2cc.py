

import argparse
import os

# import ruamel_yaml as yaml # default, but error
import ruamel.yaml as yaml

import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from transformers import BertForMaskedLM
from torchvision import transforms
from PIL import Image

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import clip

import utils

from SimVLA import SimAttacker, ImageAttacker, TextAttacker      






from dataset import paired_dataset

# score_i2t, score_t2i, t_score_i2t, t_score_t2i= 
# retrieval_eval(model, ref_model, t_model, t_ref_model, t_test_transform,
#                     data_loader, tokenizer, t_tokenizer, device, config)
def retrieval_eval(model, ref_model, t_model, t_ref_model, t_test_transform, data_loader, tokenizer, t_tokenizer, device, config):
    # test
    model.float()      # 将模型的所有参数和缓冲区转换为 float32 类型
    model.eval()
    ref_model.eval()
    t_model.float()
    t_model.eval()
    t_ref_model.eval()    


    print('Computing features for evaluation adv...')

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_attacker = ImageAttacker(images_normalize, eps=2.0/255, steps=10, step_size=0.5/255) # default: eps=2/255, steps=10, step_size=0.5/255 (按照4:1设置步长) 
    txt_attacker = TextAttacker(ref_model, tokenizer, cls=False, max_length=30, number_perturbation=1,
                                topk=10, threshold_pred_score=0.3)
    attacker = SimAttacker(model, img_attacker, txt_attacker)

    print('Prepare memory')
    num_text = len(data_loader.dataset.text)       # flickr: 5000                     
    num_image = len(data_loader.dataset.ann)       # flickr: 1000                    

    # CLIP-ViT
    t_image_feats = torch.zeros(num_image, t_model.visual.output_dim)   # [1000, 512]
    t_text_feats = torch.zeros(num_text, t_model.visual.output_dim)     # [5000, 512]

    # ALBEF
    s_image_feats = torch.zeros(num_image, config['embed_dim'])     # flickr: [1000,256]
    s_image_embeds = torch.zeros(num_image, 577, 768)               # flickr: [1000,577,768]     
    s_text_feats = torch.zeros(num_text, config['embed_dim'])       # flickr: [5000,256]        
    s_text_embeds = torch.zeros(num_text, 30, 768)                  # flickr: [5000,30,768]       
    s_text_atts = torch.zeros(num_text, 30).long()                  # flickr: [5000,30]    
    

    if args.scales is not None:
        scales = [float(itm) for itm in args.scales.split(',')] # '0.5,0.75,1.25,1.5'
        print(scales)                                           # [0.5, 0.75, 1.25, 1.5]
    else:
        scales = None

    print('Forward')
    for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(data_loader): 
        # 进来的 images 是0-1之间的，但没有经过(mean, std)正则化
        # texts_group 是batchsize个列表，每个列表5个caption. 例如 batchsize=2 时:
            # (['the man with pierced ears is wearing glasses and an orange hat', 
            #   'a man with glasses is wearing a beer can crocheted hat', 
            #   'a man with gauges and glasses is wearing a blitz hat', 
            #   'a man in an orange hat starring at something', 
            #   'a man wears an orange hat and glasses'], 
            #   ['a black and white dog is running in a grassy garden surrounded by a white fence', 
            #    'a boston terrier is running on lush green grass in front of a white fence', 
            #    'a black and white dog is running through the grass', 
            #    'a dog runs on the green grass near a wooden fence', 
            #    'a boston terrier is running in the grass'])         
        # images_ids: [0, 1]
        # text_ids_groups: ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])


        print(f'--------------------> batch:{batch_idx}/{len(data_loader)}')

        texts_ids = []
        txt2img = []
        texts = []
        for i in range(len(texts_group)):           # 应该是 batchsize 数
            texts += texts_group[i]                 # [['abc','sl'],['skdf','sa']] -> ['abc','sl','skdf','sa']: 把不同image对应的caption全放到1个列表中
            texts_ids += text_ids_groups[i]         # 列表 + 是拆成元素追加，.append() 是整体追加
            # 例如: i=1 时 text_ids_groups[1] = [15,16,17,18,19]
            #       循环结束最终 [10,11,12,13,14, 15,16,17,18,19]

            txt2img += [i]*len(text_ids_groups[i])   
            # 列表相乘就是复制整个列表中的元素，这里复制的次数 len(text_ids_groups[i]) = 5
            # 比如：['d','p']*3  = ['d', 'p', 'd', 'p', 'd', 'p']
            # 假设batchsize=2，循环结束最终 txt2img  = [0,0,0,0,0, 1,1,1,1,1]  用处？？？？？？ 标记batch中的文本


        images = images.to(device)                                                                  
        adv_images, adv_texts = attacker.attack(images, texts, txt2img, device=device,
                                                max_lemgth=30, scales=scales) 


        with torch.no_grad():
            s_adv_images_norm = images_normalize(adv_images)   # 送入模型前还得 mean,sd 标准化，adv_images 估计是 0-1 之间的张量
            adv_texts_input = tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, 
                                        return_tensors="pt").to(device)               
            # 文本转化为数字 'input_ids'
            # adv_texts_input 是字典包含 'input_ids', 'token_type_ids', 'attention_mask' 三个键  
                
            s_output_img = model.inference_image(s_adv_images_norm)                   # 对抗图片在代理模型的输出 # return {'image_feat': image_feat, 'image_embed': image_embed}
            s_output_txt = model.inference_text(adv_texts_input)                      # 对抗文本在代理模型的输出 # return {'text_feat': text_feat, 'text_embed': text_embed}
            # inference 函数是直接吃三个键的字典吗？

            # 下面是对应位置填入每个 image/caption 的嵌入
            s_text_feats[texts_ids] = s_output_txt['text_feat'].cpu().detach()
            # s_text_feats: [5000, projected_dim=256] 是 s_text_embeds 的投影浓缩版
            # 只包含 [CLS] token 的特征表示，是整个文本序列的全局表示，经过了线性投影和归一化，信息更加集中，适合用于全局性任务

            s_text_embeds[texts_ids] = s_output_txt['text_embed'].cpu().detach()
            # s_text_embeds: [5000, sequence_length=30, hidden_size=768]
            # 整个文本序列的所有 token 的嵌入表示，信息更加丰富，保留了每个 token 的上下文信息

            s_text_atts[texts_ids] = adv_texts_input.attention_mask.cpu().detach()   
            # 指示输入序列中哪些 token 是有效的，哪些是填充的
            # 如：['i like apples.'] ['we do not eat apples.']
            # tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])            

            s_image_feats[images_ids] = s_output_img['image_feat'].cpu().detach()      # s_image_feats: [1000,256]
            s_image_embeds[images_ids] = s_output_img['image_embed'].cpu().detach()    # s_image_embeds: [1000,577,768]



            ## 对抗图片/文本 送入 target 模型 CLIP ##
            t_adv_img_list = []     # 接收一个batch内  xxx  的对抗图片 
            for itm in adv_images:
                t_adv_img_list.append(t_test_transform(itm))
            t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)      # 将列表转化为 batch 格式


            t_adv_images_norm = images_normalize(t_adv_imgs)            # 送入模型前还得 mean,sd 标准化 
            # 为什么不直接使用 s_adv_images_norm？ 
            # 答：不同模型的预处理要求不同，比如transform里面resize的尺寸不同，Clip 需要 CenterCrop 而 ALBEF 不需要
            output = t_model.inference(t_adv_images_norm, adv_texts)    # Clip 目标模型
            # clip 的 .inference 只有2个键: 'text_feat', 'image_feat'; 
            # 注: albef 的 .inference 有 5 个键: 'text_feat', 'image_feat', 'text_embed', 'image_embed', 'fusion_output'
            t_image_feats[images_ids] = output['image_feat'].cpu().float().detach()    # [1000, 512]
            t_text_feats[texts_ids] = output['text_feat'].cpu().float().detach()       # [5000, 512]

                
    s_score_matrix_i2t, s_score_matrix_t2i = retrieval_score(model, s_image_feats, s_image_embeds, s_text_feats,
                                                         s_text_embeds, s_text_atts, num_image, num_text, device=device)
    # s_score_matrix_i2t: [1000,5000]; s_score_matrix_t2i: [5000,1000]
    # 代理模型 ALBEF 上面为什么这么复杂: 先用feat排序筛选128个，然后将embeds重新输入Cross-Attn，
    #     取Cls继续放入 ALBEF 的 itm_head 重新计算分数？ 
    #     -- 应该是 ALBEF 模型结构决定的，Clip 没有 multimodal encoder 简单一点，直接用俩个模态的encoder输出计算的矩阵即可


    t_sims_matrix = t_image_feats @ t_text_feats.t()  # [1000,5000] = [1000,512] @ [512,5000]
    # 目标模型 Clip 用于检索的分数计算更简单，直接用 cls_token feat 来计算分数，因为没有 multimodal encoder


    return s_score_matrix_i2t.cpu().numpy(), s_score_matrix_t2i.cpu().numpy(), \
        t_sims_matrix.cpu().numpy(), t_sims_matrix.t().cpu().numpy()


@torch.no_grad()
def retrieval_score(model, image_feats, image_embeds, text_feats, text_embeds, text_atts, num_image, num_text, device=None):
    if device is None:
        device = image_embeds.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation Direction Similarity With Bert Attack:'


    sims_matrix = image_feats @ text_feats.t() # flickr: [1000, 256] @ [5000, 256]^T = [1000, 5000]
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)     # torch.size([1000,5000])，初始都是 -100
    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)): 
        # 50 次迭代输出一次日志
        # 按行迭代，sims 是 sims_matrix 的各行，维度 [5000]
        # i 为图片索引, 把每张图片对应的分数取出来

        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)                    # k=128，topk_idx 维度 [128]
        encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device) 
        # 与接下来要处理的 128 个文本特征进行融合
        # image_embeds:[1000,577,768]; encoder_output:[128, 577,768]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        # encoder_att 维度: [128, 577]，全 1 的张量，作为图像的注意力掩码
        output = model.text_encoder(encoder_embeds=text_embeds[topk_idx].to(device), # text_embeds[topk_idx] 维度: [128, 30, 768]，表示 128 个最相似文本的嵌入
                                    attention_mask=text_atts[topk_idx].to(device),   # text_atts[topk_idx] 维度: [128, 30]，表示 128 个最相似文本的注意力掩码
                                    encoder_hidden_states=encoder_output,            # 接收图像的 embedding，这里 [128, 577,768] 
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion')                                   # 'fusion' 是文本嵌入于图像嵌入融合；普通模式是只接收输入文本，输出文本嵌入
        # 这里接收 embeds 是由 albef 的结构决定的, 参考albef的流程图
        # model.inference_text 内部也有 model.text_encoder 调用，但这里不是从头开始输入，而是直接接收文本嵌入
        # 普通字典不能使用 ".方法"，output 是一个 ModelOutput 对象，它包含多个键，该类重载了 __getattr__ 方法，使得键名可以像属性一样被访问
        # 在 fusion 模式下，text_encoder 会将这些输入通过交叉注意力机制进行融合。融合后的表示会存储在 output.last_hidden_state 中，
        # output.last_hidden_state 形状为 [128, 30, 768]，表示每个文本 token 经过图像特征增强后的表示

        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        # self.itm_head = nn.Linear(text_width, 2) 
        
        # output.last_hidden_state[:, 0, :] 维度: [128, 768]，提取每个融合后的第一个 token 的表示（通常是 [CLS] token 或类似的）
        # model.itm_head(output.last_hidden_state[:, 0, :]) 输出 [128, 2], 是一个二分类问题 
        # [:, 1] 是输出 '匹配' 这一类的分数
        # 最终 score 是一个 [128] 维的 tensor

        score_matrix_i2t[i, topk_idx] = score
        # score_matrix_i2t 维度: [1000, 5000]（假设共有 1000 张图像和 5000 个文本）
        # 表示第 i/1000 张图像与 128 个最相似文本的匹配得分填入，其余位置为初始化的 -100


    sims_matrix = sims_matrix.t()                                                 # [5000,1000]
    score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)       # [5000,1000]
    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):   # sims: [1000]
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        # image_embeds:[1000,577,768]; encoder_output:[128,577,768]

        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)  # encoder_att: [128,577]
        output = model.text_encoder(encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1).to(device), 
                                    # text_embeds: [5000,30,768] 
                                    # encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1): [128,30,768]
                                    attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
                                    encoder_hidden_states=encoder_output,                  # encoder_output:[128,577,768]
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion')
        # 为啥不直接 net.inference(images, text_inputs)['fusion_output']？可能需要前128个？
                                    
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        # output.last_hidden_state 应该是 ALBEF 融合模块的最后一个全连接层输出
        # output.last_hidden_state: [128, 30, 768], output.last_hidden_state[:, 0, :] 是 [CLS] token: [128,768]
        # score 是 [128]

        score_matrix_t2i[i, topk_idx] = score # i取值0-4999, 形状 [5000,1000]

    return score_matrix_i2t, score_matrix_t2i


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img, model_name):
# t_result = itm_eval(t_score_i2t, t_score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'CLIP_ViT')
# t_score_i2t: [1000,5000]; t_score_t2i: [5000,1000] 是分数
# data_loader.dataset.img2txt 长度1000的字典，每个键包含5个正确句子索引索引组成的列表; data_loader.dataset.txt2img，长度5000的字典，每个键包含其1个正确图片索引


    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])        # [1000]
    for index, score in enumerate(scores_i2t):   # score(一行): [5000]; t_score_i2t: [1000,5000]; index: 0-999
        inds = np.argsort(score)[::-1]           # inds: 5000个对抗caption 从大到小 排序的索引。左边的大，右边的小
        # Score
        rank = 1e20
        for i in img2txt[index]:                 
            # img2txt: 1000个键，每个键有5个caption的索引; img2txt[index] 是容量为5的索引列表
            # i 是一个数，是图片对应的正确索引，如: 遍历 [5,6,7,8,9]

            tmp = np.where(inds == i)[0][0]      
            # inds中等于正确索引的位置 # np.where 返回符合条件的索引组成的tuple，需要用[0]取出其中元素

            if tmp < rank:
                rank = tmp
        ranks[index] = rank                      # 记录5个原正确caption索引中分数最高(最左边)的位次
                                                 # ranks: [1000]

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)       # flickr: 1000 张对抗图片，仍分类正确的百分比 (原对应的5个caption得分最高的也没在前1个)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_tr1 = np.where(ranks < 1)[0]    # 返回图片索引
    after_attack_tr5 = np.where(ranks < 5)[0]
    after_attack_tr10 = np.where(ranks < 10)[0]

    
    original_rank_index_path = args.original_rank_index_path
    origin_tr1 = np.load(f'{original_rank_index_path}/{model_name}_tr1_rank_index.npy')
    origin_tr5 = np.load(f'{original_rank_index_path}/{model_name}_tr5_rank_index.npy')
    origin_tr10 = np.load(f'{original_rank_index_path}/{model_name}_tr10_rank_index.npy')

    asr_tr1 = round(100.0 * len(np.setdiff1d(origin_tr1, after_attack_tr1)) / len(origin_tr1), 2) # 在原来的分类成功的样本里，但是现在不在攻击后的成功分类集合里
    asr_tr5 = round(100.0 * len(np.setdiff1d(origin_tr5, after_attack_tr5)) / len(origin_tr5), 2)
    asr_tr10 = round(100.0 * len(np.setdiff1d(origin_tr10, after_attack_tr10)) / len(origin_tr10), 2)
    # np.setdiff1d(A, B) 在A出现，但不在后面B出现的元素


    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])           # [5000]
    for index, score in enumerate(scores_t2i):      # score 是某个 caption 的分数 [1000]; index: 0-4999
        inds = np.argsort(score)[::-1]              # [1000]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_ir1 = np.where(ranks < 1)[0]
    after_attack_ir5 = np.where(ranks < 5)[0]
    after_attack_ir10 = np.where(ranks < 10)[0]

    origin_ir1 = np.load(f'{original_rank_index_path}/{model_name}_ir1_rank_index.npy')
    origin_ir5 = np.load(f'{original_rank_index_path}/{model_name}_ir5_rank_index.npy')
    origin_ir10 = np.load(f'{original_rank_index_path}/{model_name}_ir10_rank_index.npy')

    asr_ir1 = round(100.0 * len(np.setdiff1d(origin_ir1, after_attack_ir1)) / len(origin_ir1), 2) 
    asr_ir5 = round(100.0 * len(np.setdiff1d(origin_ir5, after_attack_ir5)) / len(origin_ir5), 2)
    asr_ir10 = round(100.0 * len(np.setdiff1d(origin_ir10, after_attack_ir10)) / len(origin_ir10), 2)


    # eval_result = {'txt_r1_ASR (txt_r1)': f'{asr_tr1}({tr1})',
    #                'txt_r5_ASR (txt_r5)': f'{asr_tr5}({tr5})',
    #                'txt_r10_ASR (txt_r10)': f'{asr_tr10}({tr10})',
    #                'img_r1_ASR (img_r1)': f'{asr_ir1}({ir1})',
    #                'img_r5_ASR (img_r5)': f'{asr_ir5}({ir5})',
    #                'img_r10_ASR (img_r10)': f'{asr_ir10}({ir10})'}
    eval_result = {'txt_r1_ASR': f'{asr_tr1}',
                   'txt_r5_ASR': f'{asr_tr5}',
                   'txt_r10_ASR': f'{asr_tr10}',
                   'img_r1_ASR': f'{asr_ir1}',
                   'img_r5_ASR': f'{asr_ir5}',
                   'img_r10_ASR': f'{asr_ir10}'}
    return eval_result

def load_model(model_name, model_ckpt, text_encoder, device):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)    
    if model_name in ['ALBEF', 'TCL']:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        # config 是全局变量；text_encoder 是一个 名字 或 路径

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
        # 将状态字典的位置编码 token个数 改为 model.visual_encoder 的位置编码 token个数
        # 
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
        # m_ 可能是动量模型?
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]  # 旧键值赋值给新键名
            del state_dict[key]                        # 删除旧键名和对应的键值
    # 这个循环是为了更新状态字典 key 的名字来适应 视觉编码器模型结构
    # 例如：'bert.encoder.layer.0.attention.self.key.weight' 变为：'encoder.layer.0.attention.self.query.weight'
    
    model.load_state_dict(state_dict, strict=False) 
    # strict=False：当模型中有额外的层或者有些层未包含在 state_dict 中时，可继续加载
    
    return model, ref_model, tokenizer

def eval_asr(model, ref_model, tokenizer, t_model, t_ref_model, t_tokenizer, t_test_transform, data_loader, device, args, config):
    model = model.to(device)
    ref_model = ref_model.to(device) 

    t_model = t_model.to(device)
    t_ref_model = t_ref_model.to(device)

    print("Start eval")
    start_time = time.time()
    
    score_i2t, score_t2i, t_score_i2t, t_score_t2i= retrieval_eval(model, ref_model, t_model, t_ref_model, t_test_transform,
                                                                   data_loader, tokenizer, t_tokenizer, device, config)
    # [1000,5000] [5000,1000] [1000,5000] [5000,1000]


    t_result = itm_eval(t_score_i2t, t_score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'CLIP_ViT')
    # t_score_i2t: [5000,1000]; t_score_t2i: [1000,5000]
    # data_loader.dataset.img2txt 长度1000的字典，每个键包含5个正确句子索引索引组成的列表; data_loader.dataset.txt2img，长度5000的字典，每个键包含其1个正确图片索引

    # result = itm_eval(score_i2t, score_t2i, data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'ALBEF')

    # print('Performance on {}: \n {}'.format(args.source_model, result))
    print('source:', args.source_model)
    print('Performance on {}: \n {}'.format(args.target_model, t_result))
    

    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))    



def main(args, config):
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    
    print("Creating Source Model")
    model, ref_model, tokenizer = load_model(args.source_model, args.source_ckpt, args.source_text_encoder, device)
    # ref_model 是 BertForMaskedLM.from_pretrained(text_encoder)   
    t_model, t_ref_model, t_tokenizer = load_model(args.target_model, args.target_ckpt, args.target_text_encoder, device)
 
    #### Dataset ####
    print("Creating dataset")
   
    s_test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC), 
        transforms.ToTensor(),        
    ])
    # albef: 384

    n_px = t_model.visual.input_resolution # clip: 224
    t_test_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        # transforms.ToTensor(),
    ])
    
    test_dataset = paired_dataset(config['test_file'], s_test_transform, config['image_root'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=4, collate_fn=test_dataset.collate_fn)

    eval_asr(model, ref_model, tokenizer, t_model, t_ref_model, t_tokenizer, t_test_transform, test_loader, device, args, config)

    # 输出当前时间
    print("Current Time:", time.asctime())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=2, type=int)

# albef_retrieval_flickr.pth
# albef_retrieval_mscoco.pth
# tcl_retrieval_checkpoint_coco.pth
# tcl_retrieval_checkpoint_flickr.pth

    parser.add_argument('--source_model', default='ALBEF', type=str)
    # parser.add_argument('--source_text_encoder', default='.bert-base-uncased', type=str) # source
    parser.add_argument('--source_text_encoder', default='./checkpoints/bert-base-uncased', type=str) # 笔记本本地部署
    parser.add_argument('--source_ckpt', default='./checkpoints/albef_retrieval_flickr.pth', type=str)    
    
    parser.add_argument('--target_model', default='RN101', type=str) # 'RN101'
    # parser.add_argument('--target_text_encoder', default='bert-base-uncased', type=str) # source
    parser.add_argument('--target_text_encoder', default='./checkpoints/bert-base-uncased', type=str) # 笔记本本地部署
    parser.add_argument('--target_ckpt', default=None, type=str)    
 
    parser.add_argument('--original_rank_index_path', default='./std_eval_idx/flickr30k/')  
    parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main(args, config)


