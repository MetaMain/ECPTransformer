from unittest.mock import patch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import sys

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from transformers import AutoTokenizer 

from models.PerceiverAR_baseline import PerceiverARTransformer
from models.DoubleAttentionTransformer import DoubleAttentionTransformer
from models.sSplitAttentionTransformer import sSplitAttentionTransformer
from models.CompressedDoubleAttentionTransformer import CompressedDoubleAttentionTransformer
from models.ECPAttentionTransformer import ECPAttentionTransformer

from models.AutoRegressiveWrapper import AutoRegressiveWrapper


import time
from tqdm import tqdm
import Utils
import math
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# parsers
parser = argparse.ArgumentParser(description='ECP Transformer models')
parser.add_argument('--dataset', default='WikiText103') # options: WikiText103, PG19
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate') 
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', default=False, help='resume from checkpoint')
parser.add_argument('--gradient_accumulate_every', default=2, help='for gradient accumulation')
parser.add_argument('--test_trained_model', default=False, help='***testing only, set False to train')
parser.add_argument('--validate_every', default=5000, help='do validation after iterations')
parser.add_argument('--generate_every', default=50000, help='do eneration after iterations')
parser.add_argument('--generate_length', default=300, help='number of tokens to generate')
parser.add_argument('--net', default='ECPAttention') # options: PerceiverAR,DoubleAttention,CompressedDoubleAttention, sSplitAttention, ECPAttention 
parser.add_argument('--heads', default=6, type=int)
parser.add_argument('--layers', default=6, type=int)  # depth
parser.add_argument('--batch_size', default=8, type=int)  
parser.add_argument('--sequence_length', default=1024, type=int)  
parser.add_argument('--latent_length', default=1024, type=int) # for PerceiverAR models 
# ***for ECPAttention, set latent_len = sequence_length (to generalize the autoregresivewrapper)
parser.add_argument('--num_segments', default=4, type=int) # number of segments to split seq into
parser.add_argument('--num_iterations', default=5e5, type=int)
parser.add_argument('--dim', default=768, type=int) # embedding dimesnion
parser.add_argument('--do_word_level_modeling', default=True) # false for character level

args = parser.parse_args()
LAST_BEST_PERPLEXITY = 999
model_file_name = "trans_" + args.net + "_N"+ str(args.sequence_length) + "_H" + str(args.heads) + "_L" + str(args.layers) + "_D" + str(args.dim) + "_S" + str(args.num_segments) + ".pt"  # change this accordingly, if restoring model from a different file
tokenizer_word = AutoTokenizer.from_pretrained("bert-base-cased",truncation=True, max_length=512) # for word level modeling

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#following functions are for character level modeling-------------------
def decode_token_char(token): # convert token to character
    return str(chr(max(32, token)))

def decode_tokens_char(tokens): # convert sequence of characters to tokens
    return ''.join(list(map(decode_token_char, tokens)))
#------------------------------------------------------------------------

def decode_tokens_word(tokens): # convert token to word - for word level modeling
    return tokenizer_word.decode(tokens)

def count_parameters(model): # count number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def configure_optimizers(mymodel):
    """
    separate all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    Return the PyTorch optimizer object.
    """

    # separate out parameters that will experience regularizing weight decay
    # and those that will not
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in mymodel.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in mymodel.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    learning_rate = args.lr
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9,0.95))
    return optimizer

def compute_perplexity_huggingface(model,test_set,device,max_len=args.sequence_length):
    global LAST_BEST_PERPLEXITY
    stride = max_len//2
    encodings = test_set.data
    encodings = encodings.view(1,encodings.size(0)*encodings.size(1))
    seq_len = encodings.size(1)
    nlls = []
    prev_end_loc = 0
    count = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_len+1, seq_len+1)
        if (end_loc - begin_loc) < (max_len+1):
            continue
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(device)
        if input_ids.shape[-1] < (max_len + 1):
            continue
        count = count + 1
        with torch.no_grad():
            loss = model(input_ids)
        nlls.append(loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    avg_loss = torch.stack(nlls).mean()
    ppl = torch.exp(avg_loss)
    best_found = False
    if LAST_BEST_PERPLEXITY == 999:
        LAST_BEST_PERPLEXITY = ppl
    else:
        if ppl < LAST_BEST_PERPLEXITY:
            LAST_BEST_PERPLEXITY = ppl
            best_found = True
            # save the best model

    #print("-----------Perplexity------------- = ", ppl, "----loss=",torch.stack(nlls).mean())
    return best_found, ppl, avg_loss

def save_model(model, i, optim, fname):
    print("----------saving model-----------------")
    checkpoint_data = {
    'epoch': i,
    'state_dict': model.state_dict(),
    'optimizer': optim.state_dict()
    }
    ckpt_path = os.path.join("checkpoint/" + fname) 
    torch.save(checkpoint_data, ckpt_path)
    model.train()


def main():
    NUM_UNIQUE_TOKENS = 204   # based on transformer XL code for enwik8, for char level modeling
    if args.do_word_level_modeling == True:
        NUM_UNIQUE_TOKENS = 28996 # bert-base_cased for wikitext-103
        
    if args.net=="PerceiverAR":  # baseline
        net = PerceiverARTransformer(
            dim = args.dim, # embedding size
            num_unique_tokens = NUM_UNIQUE_TOKENS,   # 28996, for bert-base_cased for wikitext-103 
            num_layers = args.layers, 
            heads = args.heads, 
            sequence_len = args.sequence_length,
            latent_len = args.latent_length,
        ).to(device)

    if args.net=="DoubleAttention":
        net = DoubleAttentionTransformer(
            dim = args.dim, # embedding size 
            num_unique_tokens = NUM_UNIQUE_TOKENS,   # 28996, for bert-base_cased for wikitext-103 
            num_layers = args.layers, 
            heads = args.heads, 
            sequence_len = args.sequence_length,
            latent_len = args.latent_length
         ).to(device)

    if args.net=="CompressedDoubleAttention":
        compression_size = args.sequence_length//4
        net = CompressedDoubleAttentionTransformer(
            dim = args.dim, # embedding size 
            num_unique_tokens = NUM_UNIQUE_TOKENS,   
            num_layers = args.layers, 
            heads = args.heads, 
            sequence_len = args.sequence_length,
            latent_len = args.latent_length,
            compression_size = compression_size
         ).to(device)    
    if args.net=="sSplitAttention":
        net = sSplitAttentionTransformer(
            dim = args.dim, # embedding size 
            num_unique_tokens = NUM_UNIQUE_TOKENS,   
            num_layers = args.layers, 
            heads = args.heads, 
            sequence_len = args.sequence_length,
            latent_len = args.latent_length,
            num_segments = args.num_segments
         ).to(device)
    if args.net=="ECPAttention":   # context propagating segment architecture
        net = ECPAttentionTransformer(
            dim = args.dim, # embedding size 
            num_unique_tokens = NUM_UNIQUE_TOKENS,   # 28996, for bert-base_cased for wikitext-103 
            num_layers = args.layers, 
            heads = args.heads, 
            sequence_len = args.sequence_length,
            num_segments = args.num_segments
         ).to(device)

    model = AutoRegressiveWrapper(net, latent_len=args.latent_length,segment_len = args.sequence_length//args.num_segments)
 
    model.to(device)
    pcount = count_parameters(model)
    print("count of parameters in the model = ", pcount/1e6, " million")

    if args.do_word_level_modeling == True:
        if args.dataset == "WikiText103":
            train_loader, val_loader, test_loader, val_dataset, test_dataset = Utils.get_loaders_wikitext_103(tokenizer_word, args.sequence_length, args.batch_size)
        if args.dataset == "PG19":    
            train_loader, val_loader, test_loader, val_dataset, test_dataset = Utils.get_loaders_pg19(tokenizer_word, args.sequence_length, args.batch_size)
    else: # char level modeling
        # train_loader, val_loader, test_loader, val_dataset = Utils.get_loaders_enwiki8(SEQ_LENGTH, BATCH_SIZE)
        train_loader, val_loader, test_loader, val_dataset = Utils.get_loaders_enwiki8_basedon_transformerXL(args.sequence_length, args.batch_size)
        #train_loader, val_loader, test_loader, val_dataset = Utils.get_loaders_text8(SEQ_LENGTH, BATCH_SIZE)

    optim = configure_optimizers(model)
    
    #-----------------testing trained model only----------------------------
    if args.test_trained_model == True:
        model.eval()
        model.model.ff_dropout = 0
        model.model.attn_dropout = 0
        for attn, ff in model.model.layers:
            attn.attn_dropout = nn.Dropout(0)
        test_model_file_name = "D:/NLP/ECPTrainedModels/TransECP_1048_768_6_6_4Seg_Perplex_21_4_500K.pt"  # change this accordingly
        checkpoint_data = torch.load(test_model_file_name)
        model.load_state_dict(checkpoint_data['state_dict'])
        optim.load_state_dict(checkpoint_data['optimizer'])
        ret, perplexity, loss = compute_perplexity_huggingface(model,test_dataset,device)
        print('Test model perplexity =', perplexity)
        return
    #--------------------end testing trained model only------------------
    
    new_learning_rate = args.lr   # if learning rate needs to be changed when resuming
    if args.resume == False:
        start = 0
    else:
        restore_model_file_name = "checkpoint/" + model_file_name  # change this accordingly
        checkpoint_data = torch.load(restore_model_file_name)
        model.load_state_dict(checkpoint_data['state_dict'])
        optim.load_state_dict(checkpoint_data['optimizer'])
        for param_group in optim.param_groups:  # if lr needs to be changed
            param_group['lr'] = new_learning_rate
        start = checkpoint_data['epoch']

    #-------------------training loop-------------------------------
    pbar = tqdm(int(args.num_iterations))
    for i in tqdm(range(start,int(args.num_iterations)), mininterval = 10., desc = 'training'):
        model.train()
        total_loss = 0
        for __ in range(args.gradient_accumulate_every):
            loss = model(next(train_loader))
            loss.backward()
        if (i%100 == 0):
            #print(f'training loss: {loss.item()} -- iteration = {i}')
            pbar.set_postfix({"training loss": f"{loss.item():.4f}","iteration": i, "last best perplexity": LAST_BEST_PERPLEXITY})

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        #------------------step learning rate adjustment----------------------
        if (i+1) % 100000 == 0:
            for param_group in optim.param_groups:  # if lr needs to be changed
                param_group['lr'] = args.lr/2
                args.lr = args.lr/2
        #------------------end step learning rate adjustment---------------------
        #------------------------do validation-----------------------------------
        if (i% args.validate_every == 0) and (args.do_word_level_modeling == True):
            ret, perplexity, loss = compute_perplexity_huggingface(model,test_dataset,device)
            if ret == True: # save best model
                pbar.set_postfix({"saving best model - perplexity":perplexity})
                save_model(model,i,optim,model_file_name) # save best model
            pbar.set_postfix({"val loss": f"{loss.item():.4f}","iteration": i,"perplexity" :f"{perplexity:.4f}"})
        #--------------------------for character level modeling------------------------
        if ((i+0) % args.validate_every == 0) and (args.do_word_level_modeling == False):
           model.eval()
           val_count = 1000  # number of validations to compute average BPC
           total_loss = 0
           with torch.no_grad():
               for v in range(0,val_count):
                   loss = model(next(test_loader))
                   total_loss += loss.item()
                   loss_m = loss.mean()
               print(f'----------validation loss: {total_loss/val_count}')
               print(f'Perplexity : {math.exp(total_loss/val_count)}, BPC: {total_loss/val_count*np.log2(2.7173)}')
        #----------------------------end do validation--------------------------
            
        #---------------------------do generation-------------------------------
        if args.generate_every == 0:  # see if model is generating in a meaningful way
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            if args.do_word_level_modeling == True:
                input_start_sequence = decode_tokens_word(inp)
            else:
                input_start_sequence = decode_tokens_char(inp)
            print("----------prompt input------------------")
            print(f'%s \n\n', (input_start_sequence))
            print("----------end of prompt input-----------")
            sample = model.generate(inp, args.generate_length)
            if args.do_word_level_modeling == True:
                output_str = decode_tokens_word(sample)
            else:
                output_str = decode_tokens_char(sample)
            print("----------generated output-------------")
            print(output_str)
            print("----------end generated output---------")
        
        model.train()
        #----------------------------end do generation---------------------------
        
if __name__ == "__main__":
    sys.exit(int(main() or 0))


