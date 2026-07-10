

import numpy as np 
import torch
import torch.nn as nn

import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

beta = 16 

num_samples = 4
gamma = 0.1

class SimAttacker():
    def __init__(self, model, img_attacker, txt_attacker):
        self.model=model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker

        self.batch_size = 32
        self.max_length = 30
        
    def attack(self, imgs, txts, txt2img, device='cpu', max_length=30, scales=[0.5,0.75,1.25,1.5], **kwargs):
    
        with torch.no_grad():
            s_imgs = self.get_noise_imgs(imgs, num_samples=num_samples, beta=beta, eps=2/255, device=device)

            origin_img_output = self.model.inference_image(self.img_attacker.normalization(s_imgs))

            # img_supervisions = origin_img_output['image_feat']
            img_supervisions = origin_img_output['image_embed']

            batch_size = imgs.shape[0]
            num_scales = num_samples + 1  
            img_supervisions = img_supervisions.view(num_scales, batch_size, *img_supervisions.shape[1:])
            img_supervisions = img_supervisions.permute(1, 0, *range(2, img_supervisions.dim()))
            img_supervisions_avg = img_supervisions.mean(dim=1)

            img_supervisions_avg = img_supervisions_avg[txt2img] 
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, txt2img, img_embeds=img_supervisions_avg)
    
        aug_adv_txts = self.del_text(adv_txts, img_embeds=img_supervisions_avg, K=1) # 1


        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(aug_adv_txts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_embed']
            
        adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img, device, 
                                                       scales=scales, txt_embeds = txt_supervisions)
                            
        return adv_imgs, adv_txts


    def get_noise_imgs(self, imgs, num_samples=4, beta=1.0, eps=1.0, device='cuda'):
        batch_size = imgs.shape[0]
        ori_shape = imgs.shape[1:]  # [3, H, W]

        noise_range = (-beta * eps, beta * eps)

        result = []

        for sample_idx in range(num_samples):
            for img_idx in range(batch_size):
                img = imgs[img_idx].unsqueeze(0)

                noisy_img = img + torch.from_numpy(np.random.uniform(noise_range[0], noise_range[1], img.shape)).float().to(device)

                result.append(noisy_img)

        return torch.cat([imgs] + result, dim=0)  # [batch_size * (num_samples + 1), 3, H, W]
    
    def get_scaled_imgs(self, imgs, scales=[0.5,0.75,1.25,1.5], device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio*ori_shape[0]), 
                                  int(ratio*ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            
            reversed_imgs = reverse_transform(scaled_imgs)
            
            result.append(reversed_imgs)
        
        return torch.cat([imgs,]+result, 0)
    
    def del_text(self, adv_txts, img_embeds, K=1):
        aug_adv_txts = []
        
        for i, adv_text in enumerate(adv_txts): 

            words = adv_text.split()
            num_words = len(words)

            if K >= num_words:
                aug_adv_txts.append(adv_text)
                continue  

            important_scores = self.get_important_scores(
                adv_text, self.model, img_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            bottom_k_indexes = [list_of_index[-(j+1)][0] for j in range(K)]

            indices_to_remove = sorted(bottom_k_indexes, reverse=True)
            for idx in indices_to_remove:
                del words[idx]

            aug_adv_txt = ' '.join(words)

            aug_adv_txts.append(aug_adv_txt)

        return aug_adv_txts

    


    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):

        device = origin_embeds.device

        masked_words = self._get_masked(text)                      
        masked_texts = [' '.join(words) for words in masked_words]  

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):    

            masked_text_input = self.txt_attacker.tokenizer(masked_texts[i:i+batch_size], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
              
                    
            masked_output = net.inference_text(masked_text_input) 

            masked_embed = masked_output['text_embed'].detach() 



            masked_embeds.append(masked_embed)

        masked_embeds = torch.cat(masked_embeds, dim=0)   

        criterion = torch.nn.KLDivLoss(reduction='none') 

        import_scores = criterion(masked_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1) 
    
    def _get_masked(self, text):

        words = text.split(' ')  
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:]) 
        return masked_words 
    

class ImageAttacker():
    def __init__(self, normalization, eps=2/255, steps=10, step_size=0.5/255):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps 
        self.step_size = step_size 

    def get_scaled_imgs(self, imgs, scales=[0.5,0.75,1.25,1.5], device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio*ori_shape[0]), 
                                  int(ratio*ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            
            reversed_imgs = reverse_transform(scaled_imgs)
            
            result.append(reversed_imgs)
        
        return torch.cat([imgs,]+result, 0)
    

    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img):  


        device = adv_imgs_embeds.device   # [batchsize, 256]

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T           # [batchsize, 5*batchsize] = [batchsize, 256] * [5*batchsize, 256]^T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)   # [batchsize, 5*batchsize]
        
        for i in range(len(txt2img)):                             
            it_labels[txt2img[i], i]=1                            
        
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        
        return loss
    
   

    def txt_guided_attack(self, model, imgs, txt2img, device, scales=None, txt_embeds=None):
        model.eval()
        
        b, c, h, w = imgs.shape
        
        scales_num = num_samples +1 # = scales_num + 1

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        # adv_imgs = imgs.detach() 
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)


        grad_cur = torch.zeros([scales_num, b, c, h, w]).to(device)
        grad_pgia = torch.zeros([scales_num, b, c, h, w]).to(device)

        decay = 1.0
        mom = 0

        for step in range(self.steps): 
            # adv_imgs.requires_grad_()
            scaled_imgs = self.get_noise_imgs(adv_imgs, num_samples=num_samples, beta=beta, eps=self.eps, device=device)   # torch.tensor([batch_size*len(scale), 3, 224, 224])  
            
            total_grad = torch.zeros_like(adv_imgs)  
            for i in range(scales_num):  

                current_scaled_img = scaled_imgs[i*b:(i+1)*b]  

                current_scaled_img = current_scaled_img + gamma * self.eps * grad_pgia[i]


                if current_scaled_img.grad is not None:
                    current_scaled_img.grad.zero_()

                current_scaled_img.requires_grad = True  

                current_embeds = model.inference_image(self.normalization(current_scaled_img))['image_feat']
                loss_item = self.loss_func(current_embeds, txt_embeds, txt2img)

                grads = torch.autograd.grad(loss_item, current_scaled_img, retain_graph=False)[0]


                grad_cur[i] = grads
                total_grad += grads


            grad_pgia = ((grad_cur / torch.mean(torch.abs(grad_cur), (2, 3, 4), keepdim=True)).detach() - grad_pgia)
            # [scales_num, b, c, h, w]

            final_grad = total_grad / scales_num

            final_grad = final_grad / torch.mean(torch.abs(final_grad), dim=(1, 2, 3), keepdim=True)

            mom = decay * mom + final_grad 
            perturbation = self.step_size * mom.sign()

            adv_imgs = adv_imgs.detach() + perturbation 
            adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        
        return adv_imgs


    def get_noise_imgs(self, imgs, num_samples=4, beta=1.0, eps=1.0, device='cuda'):

        batch_size = imgs.shape[0]
        ori_shape = imgs.shape[1:]  # [3, H, W]

        noise_range = (-beta * eps, beta * eps)

        result = []

        for sample_idx in range(num_samples):
            for img_idx in range(batch_size):
                img = imgs[img_idx].unsqueeze(0)

                noisy_img = img + torch.from_numpy(np.random.uniform(noise_range[0], noise_range[1], img.shape)).float().to(device)

                result.append(noisy_img)

        return torch.cat([imgs] + result, dim=0) 





    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio*ori_shape[0]), 
                                  int(ratio*ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            
            reversed_imgs = reverse_transform(scaled_imgs)
            
            result.append(reversed_imgs)
        
        return torch.cat([imgs,]+result, 0)



filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)
    

class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10, threshold_pred_score=0.3, batch_size=32):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls

    def img_guided_attack(self, net, texts, txt2img, img_embeds = None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            # origin_embeds = origin_output['text_feat'][:, 0, :].detach()
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            # origin_embeds = origin_output['text_feat'].flatten(1).detach()
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, img_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)


                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    # replace_embeds = replace_output['text_feat'][:, 0, :]
                    replace_embeds = replace_output['text_embed'][:, 0, :]

                else:
                    # replace_embeds = replace_output['text_feat'].flatten(1)
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = self.loss_func(replace_embeds, img_embeds, i)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1) 
        loss = loss_TaIcpos
        return loss

 
    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 1  

        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i+batch_size], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            # if self.cls:
            #     masked_embed = masked_output['text_feat'][:, 0, :].detach()
            # else:
            #     masked_embed = masked_output['text_feat'].flatten(1).detach()
            if self.cls:
                masked_embed = masked_output['text_embed'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_embed'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)



def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
