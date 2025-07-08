import json
import csv
import re
import os
import torch

from torch.utils.data import Sampler
import torch.distributed as dist
import random
from collections import defaultdict
import math
import random
import torchaudio
import torch.nn as nn
import numpy as np
import pandas as pd
from jiwer import wer
from tqdm.auto import tqdm
import torch.nn.functional as F
from IPython.display import Audio
from functools import partial
from torch.utils.data import Dataset
from transformers import Wav2Vec2Model
from torch.utils.data import DataLoader
# AdamW is best optimizer
from torch.optim import AdamW
from transformers import get_scheduler
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
warnings.filterwarnings("ignore")



def get_latest_checkpoint(base_dir, prefix = "weights_", model_prefix = "multilingual_asr_model_"):
    
    """
    Finds the latest checkpoint directory based on numeric ranges in the names.
    Returns the last_end_index and full path of the checkpoint.
    """

    if not os.path.exists(base_dir):
        return 0, None

    checkpoint_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith(prefix)])
    
    if not checkpoint_dirs:
        return 0, None

    # Extract numeric start and end indices
    max_end = 0
    latest_ckpt_path = None
    for d in checkpoint_dirs:
        match = re.match(rf"{prefix}(\d+)_(\d+)", d)
        if match:
            start, end = map(int, match.groups())
            if end > max_end:
                max_end = end
                latest_ckpt_path = os.path.join(base_dir, d, f"{model_prefix}{start}_{end}.pt")
                
    latest_ckpt_path if latest_ckpt_path and os.path.exists(latest_ckpt_path) else None

    if latest_ckpt_path and os.path.exists(latest_ckpt_path):
        return max_end, latest_ckpt_path

    else:
        return max_end, None


    # For the first time i have to use this check points of the epoch 2 it is the multilingual vocab model from this i have to remove the weights of the projector layer and the multilingual vocab layer.
    
    # return 0, '/scratch/IITB/ai-at-ieor/23m1508/23m1508_backup/checkpoints_mtp/w_till_epoch_1/w_till_epoch_2/multilingual_asr_model_1130000_1150000.pt'



# Declare globals outside
new_start = None
new_end = None


def generate_config(base_dir, csv_path, model_id, loss_dir_base, chunk_size, batch_size,  epochs, lr, freeze_layers):
    
    global new_start, new_end
    
    # Get latest state
    latest_end, latest_checkpoint = get_latest_checkpoint(base_dir)
    new_start = latest_end
    new_end = new_start + chunk_size

    new_checkpoint_subdir = f"weights_{new_start}_{new_end}"
    new_checkpoint_path = os.path.join(base_dir, new_checkpoint_subdir)
    os.makedirs(new_checkpoint_path, exist_ok = True)

    config = {
                "CSV_DATA_PATH": csv_path,
                "BASE_MODEL_ID": model_id,
                "start_data": new_start,
                "end_data": new_end,
                "device": 1,
                "BATCH_SIZE": batch_size,
                "EPOCHS": epochs,
                "LR": lr,
                "Number_of_first_layers_freeze_transofrmer": freeze_layers,
                "num_warmup_steps": 200,
                "loss_csv_saving_path": loss_dir_base,
                "loss_csv_name": f"loss_{new_start}_{new_end}.csv",
                "prev_checkpoint_dir": latest_checkpoint if latest_checkpoint else "",
                "new_checkpoint_dir": new_checkpoint_path,
                "model_name": f"multilingual_asr_model_{new_start}_{new_end}",
                "load_from_prev": latest_checkpoint is not None,
                "resume_training": False
            }

    print('The new start: ', new_start, 'The new end: ', new_end)
    return config
    

    
config = generate_config(
                            base_dir = "/scratch/IITB/ai-at-ieor/23m1508/23m1508_backup/checkpoints_points_approach_2",
                            
                            csv_path = "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/data/scratch_balanced_shuffled_data.csv",

                            # https://huggingface.co/facebook/wav2vec2-xls-r-1b
                            # facebook/wav2vec2-xls-r-1b, # facebook/wav2vec2-xls-r-2b
                            # using the 2 billion parameter model
                            model_id = "facebook/wav2vec2-xls-r-2b",
                            
                            # Saving the loss files in the scratch
                            loss_dir_base = "/scratch/IITB/ai-at-ieor/23m1508/23m1508_backup/checkpoints_points_approach_2/model_loss_files_approach2",
                            
                            chunk_size = 20000,
                            batch_size = 5,
                            epochs = 1,
                            lr = 1e-5,
                            freeze_layers = 5
                        )


print('The configration is :', config)


device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")


# ID,Text,Path,class
final_data = pd.read_csv(config['CSV_DATA_PATH'])


final_data = final_data.loc[config['start_data'] : config['end_data'], :]


def lowercase_all_letters(text):
    return ''.join([c.lower() if c.isalpha() else c for c in text])

final_data['Text'] = final_data['Text'].apply(lowercase_all_letters)


def clean_text(text):
    characters_to_remove = r'[,\?\.\!\-\;\:\%\'\`\{}()@#$%^&*\+\[\]\_｡]+'
    cleaned_text = re.sub(characters_to_remove, '', text)
    # cleaned_text = ''.join([c.lower() if c.isascii() else c for c in cleaned_text]) + ' '
    return cleaned_text


final_data['Text'] = final_data['Text'].apply(clean_text)
final_data = final_data.reset_index(drop = True)


print('The shape of the final data is :', final_data.shape)
print('The class distribution is given below: ')
print(final_data.loc[:, 'class'].value_counts())



# main the order as per the lang2id
vocab_paths = {
                "english" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/eng_vocab.json",
                "korean" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/kor_vocab.json",
                "japanese" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/jap_vocab.json",
                "italian" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/ita_vocab.json",
                "russian" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/rus_vocab.json",
                "portuguese" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/por_vocab.json",
                "spanish" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/spa_vocab.json",
                "vietnamese" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/vie_vocab.json",
                "french" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/fre_vocab.json",
                "thai" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/tha_vocab.json",
                "german" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/ger_vocab.json"
                }



lang2id = {    
            "english" :  0,
            "korean" :   1,
            "japanese" : 2,
            "italian" :  3,
            "russian" :  4,
            "portuguese":5,
            "spanish" :  6,
            "vietnamese":7,
            "french" :   8,
            "thai" :     9,
            "german" :   10
            }



token_vocab_map = {}  # To hold vocab dictionaries
vocab_sizes = [0] * len(lang2id)  # Indexed by lang2id


for lang, idx in lang2id.items():
    vocab_path = vocab_paths[lang]
    with open(vocab_path, "r", encoding="utf-8") as vf:
        lang_vocab = json.load(vf)
        token_vocab_map[lang] = lang_vocab
        vocab_sizes[idx] = len(lang_vocab)


print("Loaded vocab sizes:", vocab_sizes)
# print("Available language vocabs:", token_vocab_map)


# 1
eng_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['english'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")
# 2
fre_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['french'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")
# 3
ger_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['german'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")
# 4
ita_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['italian'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")
# 5
jap_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['japanese'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")
# 6
kor_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['korean'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")
# 7
por_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['portuguese'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")
# 8
rus_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['russian'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")
# 9
spa_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['spanish'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")
# 10
tha_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['thai'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")
# 11
vie_tokenizer = Wav2Vec2CTCTokenizer(vocab_paths['vietnamese'], bos_token = "<s>", eos_token = "</s>",
                                            unk_token = "<unk>", pad_token = "<pad>", word_delimiter_token = "|")



feature_extractor = Wav2Vec2FeatureExtractor(feature_size = 1, 
                                             sampling_rate = 16000, 
                                             padding_value = 0.0, 
                                             do_normalize = True, 
                                             return_attention_mask = True)


tokenizers = {
                "english": eng_tokenizer,   #0
                "korean": kor_tokenizer,    #1
                "japanese": jap_tokenizer,  #2
                "italian": ita_tokenizer,   #3
                "russian": rus_tokenizer,   #4
                "portuguese": por_tokenizer,#5
                "spanish": spa_tokenizer,   #6
                "vietnamese": vie_tokenizer,#7
                "french": fre_tokenizer,    #8
                "thai": tha_tokenizer,      #9
                "german": ger_tokenizer    #10
            }


# Dataset class for multi-head CTC
class MultiLangSpeechDataset(Dataset):
    def __init__(self, df, tokenizers: dict, feature_extractor, lang2id: dict):
        
        self.df = df.reset_index(drop = True)
        self.tokenizers = tokenizers
        self.feature_extractor = feature_extractor
        self.lang2id = lang2id  # e.g., {'eng': 0, 'fre': 1, ..., 'vie': 10}

    def __len__(self):
        return len(self.df)


    # ID,Text,Path,class --> the data set have these columns
    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        
        waveform, sample_rate = torchaudio.load(row["Path"])
        text = row["Text"]
        lang = row["class"]

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        waveform = waveform.squeeze()

        # Feature extraction
        input_values = self.feature_extractor(
                        waveform, sampling_rate = 16000, return_tensors = "pt"
                    )["input_values"].squeeze(0)

        # Label tokenization using correct tokenizer
        tokenizer = self.tokenizers[lang]
        labels = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

        return {
                    "input_values": input_values,
                    "labels":       labels,
                    "lang_id":      self.lang2id[lang],  # for selecting the CTC head
                }


speech_dataset = MultiLangSpeechDataset(
                                        df = final_data,
                                        tokenizers = tokenizers,
                                        feature_extractor = feature_extractor,
                                        lang2id = lang2id
                                        )



print("The length of the speech dataset", len(speech_dataset))



def collate_function(batch, feature_extractor, padding = True):
    # Extract input values
    
    b_X = [{"input_values": sample["input_values"]} for sample in batch]
    
    # Pad audio inputs
    features = feature_extractor.pad(
                                        b_X,
                                        padding = padding,
                                        return_tensors = "pt"
                                    )

    # Extract labels and lang_ids
    all_labels = []
    all_langs = []
    
    for sample in batch:
        all_labels.append(sample["labels"])
        all_langs.append(sample["lang_id"])

    # Pad labels manually with -100 (used for CTC loss)
    padded_labels = pad_sequence(all_labels, batch_first = True, padding_value = -100)

    features["labels"] = padded_labels
    features["lang_id"] = torch.tensor(all_langs, dtype = torch.long)

    return features

  

collate_fn = partial(collate_function, 
                     feature_extractor = feature_extractor, 
                     padding = True)


batch_size = config['BATCH_SIZE']
epochs = config['EPOCHS']
lr = config['LR']


print('The batch size is :', batch_size)


train_dataloader = DataLoader(
                                speech_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                num_workers=4
                            )


# for batch in train_dataloader:
    
#     print("The batch is:", batch)
#     print("==" * 50)
    
#     print("Input values shape:", batch["input_values"].shape)
#     print("Labels shape:", batch["labels"].shape)
#     print("The Labels Is:", batch["labels"])
#     print("Lang IDs:", batch["lang_id"])
    
#     # Additional checks (optional)
#     assert batch["input_values"].dim() == 2, "input_values should be 2D"
#     assert batch["labels"].dim() == 2, "labels should be 2D"
#     assert batch["lang_id"].dim() == 1, "lang_id should be 1D"
#     assert batch["input_values"].size(0) == batch["labels"].size(0) == batch["lang_id"].size(0), "Batch size mismatch"
    
#     print("Batch is valid ✅")
#     break  # only check the first batch




# print('Before Exiting the code')
# exit()
#Till there is code is working fine
##############################################################





# Loaidng the pretrained model
model = Wav2Vec2Model.from_pretrained(config['BASE_MODEL_ID'])




# # Use the updated base model for the first time for this new model multiple ctc head

# checkpoint = torch.load("/scratch/IITB/ai-at-ieor/23m1508/23m1508_backup/checkpoint_points/updated_base_model_weights/updated_base_wav2vec2_model_1.pt", map_location='cpu')

# missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict = False)


# print("Missing keys:", missing)
# print("Unexpected keys:", unexpected)



    
# Freeze feature extractor (same as model.freeze_feature_encoder())
for param in model.feature_extractor.parameters():
    param.requires_grad = False


# Freeze feature projection (optional)
for param in model.feature_projection.parameters():
    param.requires_grad = False
    

# Freeze first transformer layers (out of 48 for base model)
if config['Number_of_first_layers_freeze_transofrmer'] is not None:
    
    print('The number of layers to freeze is :', config['Number_of_first_layers_freeze_transofrmer'])
    
    for i in range(config['Number_of_first_layers_freeze_transofrmer']):
        for param in model.encoder.layers[i].parameters():
            param.requires_grad = False
        



# # Freeze all parameters in the model    
# for param in model.parameters():
#     param.requires_grad = False

# This is a standard property of PyTorch: only the parameters involved in the forward pass will receive gradients during backprop.




# ==================== MultiHeadCTCModel ====================
class MultiHeadCTCModel(nn.Module):
    def __init__(self, model, vocab_sizes):
        super().__init__()
        self.wav2vec2 = model
        
        self.dropout = nn.Dropout(0.1)
        
        self.ctc_heads = nn.ModuleList([
                                            nn.Linear(1920, vocab_size) for vocab_size in vocab_sizes
                                        ])

    def forward(self, input_values, attention_mask):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        
        shared_output = self.dropout(outputs.last_hidden_state)
        
        return shared_output  # Only return encoder output

# ==================== MultiHeadCTCModel ====================


multilingual_asr_model = MultiHeadCTCModel(model, vocab_sizes = vocab_sizes)


trainable_params = sum(p.numel() for p in multilingual_asr_model.parameters() if p.requires_grad)

total_params = sum(p.numel() for p in multilingual_asr_model.parameters())


print(f"Trainable: {trainable_params} / Total: {total_params}")


optimizer = AdamW(multilingual_asr_model.parameters(), lr = lr)

num_training_steps = epochs * len(train_dataloader)

print('The number of training steps is :', num_training_steps)

lr_scheduler = get_scheduler(
                            "linear",
                            optimizer = optimizer,
                            num_warmup_steps = config['num_warmup_steps'],
                            num_training_steps = num_training_steps
                            )


scaler = torch.amp.GradScaler()

multilingual_asr_model = multilingual_asr_model.train()



csv_dir = config['loss_csv_saving_path']
csv_file_name = config['loss_csv_name']
csv_file_path = os.path.join(csv_dir, csv_file_name)
# Ensure the directory exists
os.makedirs(csv_dir, exist_ok=True)

# If the CSV file doesn't exist, create it with headers
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["step", "epoch", "loss", "avg_wer"] + [f"wer_lang_{i}" for i in range(len(lang2id))])
        writer.writeheader()



new_checkpoint_dir = config['new_checkpoint_dir']

os.makedirs(new_checkpoint_dir, exist_ok=True)

start_epoch = 0
curr_best_loss = 1e9

resume_training = config['resume_training']  # set this to False to start fresh

model_name = config['model_name']

new_checkpoint_path = os.path.join(new_checkpoint_dir, model_name + ".pt")


if config['load_from_prev']:
    checkpoint_path = config['prev_checkpoint_dir']
    checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = False)
    
    multilingual_asr_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_state = checkpoint['optimizer_state_dict']
    scheduler_state = checkpoint['scheduler_state_dict']
    start_epoch = checkpoint['epoch']
    
    curr_best_loss = checkpoint.get('best_loss', curr_best_loss)

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Resuming at epoch {start_epoch}, best loss: {curr_best_loss:.4f}")
    print('Loaded form the previous model: ')
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Trained till to epoch {start_epoch} with best loss {curr_best_loss:.4f}")
    


# Move model to device and wrap with DDP
multilingual_asr_model.to(device)


# Move optimizer state to the correct device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)


# Note that it is WER for each sample
def compute_metrics(label_tensor, pred_tensor, lang_id):
    
    lang_name = [lang for lang, idx in lang2id.items() if idx == lang_id]
    
    if not lang_name:
        print(f"Warning: lang_id {lang_id} not found in lang2id.")
        return 1.0
    
    lang_name = lang_name[0]
    
    tokenizer = tokenizers[lang_name]

    # Clean up labels
    label_tensor[label_tensor == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_tensor, skip_special_tokens=True)[0]
    
    label_str = tokenizer.batch_decode(label_tensor, group_tokens = False, skip_special_tokens = True)[0]
    
    
    # print('The pred str is :', pred_str)
    # print('The label str is :', label_str)

    if label_str.strip() == "":
        return 1.0

    return wer([label_str], [pred_str])


from collections import defaultdict


curr_best_loss = 1e9


ctc_loss_fn = nn.CTCLoss(blank = 0, zero_infinity = True, reduction = 'mean')

# Training loop
# for n in tqdm(range(start_epoch, start_epoch + 1)):
for n in range(start_epoch, start_epoch + 1):

    print(f"\n=== Epoch {n} ===")
    
    losses = []
    wers = []

    total_number_batch = len(train_dataloader)
    
    multilingual_asr_model.train()

    # for step, batch in enumerate(tqdm(train_dataloader)):
    for step, batch in enumerate(train_dataloader):
        
        optimizer.zero_grad()
        batch = {k: v.to(device, non_blocking = True) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled = True):
            shared_output = multilingual_asr_model(
                batch["input_values"], attention_mask = batch.get("attention_mask")
            )

            input_lengths = torch.full(
                (shared_output.size(0),), shared_output.size(1), dtype=torch.long, device=device
            )
            
            label_lengths = (batch["labels"] != -100).sum(dim=1)

            lang_ids = batch["lang_id"]
            # print("the lang ids is: ", lang_ids)
            
            
            loss_accum = 0.0
            total_loss = 0.0
            
            head_usage = defaultdict(int)
            lang_wers = defaultdict(list)


            #====================================
            #  compute_metrics(label_tensor, pred_tensor, lang_id)
            ######################################################
            for i in range(shared_output.size(0)):
                
                head_idx = batch["lang_id"][i].item()
                # print('the head index chosen', head_idx)
                head_usage[head_idx] += 1
                
                # use module since it ddp traing
                head = multilingual_asr_model.ctc_heads[head_idx]  # Only selected head
                # print('the head is (layer model)', head)
                
                
                log_probs = F.log_softmax(head(shared_output[i:i+1]), dim = -1).permute(1, 0, 2)
                
                loss = ctc_loss_fn(
                                        log_probs,
                                        batch["labels"][i:i+1],
                                        input_lengths[i:i+1],
                                        label_lengths[i:i+1]
                                    )
                
                total_loss += loss
                
                # Decode prediction and label for WER (single sample)
                pred = torch.argmax(log_probs, dim = -1).transpose(0, 1)
                
                label = batch["labels"][i:i+1]
                labels_for_metrics = label.clone()
                # Compute WER directly
                wer_val = compute_metrics(labels_for_metrics, pred, head_idx)
                lang_wers[head_idx].append(wer_val)
            ###########################################################

            # Average loss across samples

        total_loss = total_loss / shared_output.size(0)
        losses.append(total_loss.item())
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        lr_scheduler.step()
        scaler.update()

        
        all_wers = [wer for wer_list in lang_wers.values() for wer in wer_list]
        average_wer = sum(all_wers) / len(all_wers)

        # this is the average WER for the batch
        wers.append(average_wer)
        
        # Prepare row for CSV logging
        wer_row = {f"wer_lang_{i}": np.mean(lang_wers[i]) if i in lang_wers else None for i in range(len(lang2id))}
        
        row = {
                    "step": step,
                    "epoch": n + 1,
                    "loss": total_loss.item(),
                    "avg_wer": average_wer,
                    **wer_row
                }
        print('the head usage is', head_usage)
        
        with open(csv_file_path, mode = 'a', newline = '') as file:
            writer = csv.DictWriter(file, fieldnames = row.keys())
            writer.writerow(row)
        
        
        # Explicit memory cleanup
        del batch, log_probs, input_lengths, loss
        torch.cuda.empty_cache()

        print(f"[Step {step}] Loss: {total_loss.item():.4f}, Avg WER: {average_wer:.4f}")
        
        print("==" * 50)

    # Average the loss and WER across GPUs
    avg_loss_tensor = torch.tensor(np.mean(losses), device = device)
    avg_wer_tensor = torch.tensor(np.mean(wers), device = device)

    result = {"loss": avg_loss_tensor, "wer": avg_wer_tensor}
    print(f"\nEPOCH {n + 1} SUMMARY: Loss = {result['loss']:.4f}, WER = {result['wer']:.4f}")


    print("EPOCH: ", n + 1)
    print(result)
    print('=' * 100)
    
    
    if result["loss"] < curr_best_loss:
        print(f"New best model found at epoch {n + 1} with loss {result['loss']:.4f}, saving...")
        curr_best_loss = result["loss"]

    # Save latest checkpoint
    checkpoint = {
                    'epoch': n + 1,
                    'model_state_dict': multilingual_asr_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': result['loss'],
                    'best_loss': curr_best_loss
                    }

    torch.save(checkpoint, new_checkpoint_path)
    print('Finally saving the model')


# Remove the previous checkpoint if configured
if config['load_from_prev'] and os.path.exists(config['prev_checkpoint_dir']):
    try:
        os.remove(config['prev_checkpoint_dir'])
        print(f"Removed previous checkpoint: {config['prev_checkpoint_dir']}")
    except Exception as e:
        print(f"Error removing previous checkpoint: {e}")
        
        


print(f'Training completed for the {new_start} to {new_end} chunk of data')
print("==" * 50)



            
            
            
            # # batched in gropby langids
            # #====================================
            # for lang_id in unique_lang_ids:
                
            #     #get indices for this lang_id
            #     indices = (lang_ids == lang_id).nonzero(as_tuple=True)[0]
                

            #     print('the indices', indices)
                
            #     if len(indices) == 0:
            #         continue
                
            #     # Slice batch for this lang
            #     lang_shared_out = shared_output[indices]
                
            #     print('the shape of the lang_shared_out', lang_shared_out.shape)
                
            #     lang_labels = batch["labels"][indices]

            #     print('the shape of the lang_labels', lang_labels.shape)
            #     lang_input_lens = input_lengths[indices]
                

            #     print('the length input length is', lang_input_lens)
                
            #     lang_label_lens = label_lengths[indices]
                
            #     print('the lang_label_lens', lang_label_lens)
                    
            #     head = multilingual_asr_model.ctc_heads[lang_id.item()]
                
            #     # Forward pass through head
            #     log_probs = F.log_softmax(head(lang_shared_out), dim=-1).transpose(0, 1)  # SInce it is batched
                
            #     # Loss
            #     loss = ctc_loss_fn(log_probs, lang_labels, lang_input_lens, lang_label_lens)
            #     total_loss += loss
                
            #     print('the lenb of the indices is', len(indices))
                
            #     pred = torch.argmax(log_probs, dim = -1).transpose(0, 1)
                
                
            #     for i, idx in enumerate(indices):
            #         label = batch["labels"][idx:idx+1]
            #         pred_i = pred[i:i+1]
            #         wer_val = compute_metrics(label.clone(), pred_i, lang_id.item())
            #         lang_wers[lang_id.item()].append(wer_val)





            #====================================
            #  compute_metrics(label_tensor, pred_tensor, lang_id)
            ######################################################
            # for i in range(shared_output.size(0)):
                
            #     head_idx = batch["lang_id"][i].item()
                
                
            #     # use module since it ddp traing
            #     head = multilingual_asr_model.ctc_heads[head_idx]  # Only selected head
                
            #     log_probs = F.log_softmax(head(shared_output[i:i+1]), dim = -1).permute(1, 0, 2)
                
            #     loss = ctc_loss_fn(
            #                             log_probs,
            #                             batch["labels"][i:i+1],
            #                             input_lengths[i:i+1],
            #                             label_lengths[i:i+1]
            #                         )
                
            #     total_loss += loss
                
            #     # Decode prediction and label for WER (single sample)
            #     pred = torch.argmax(log_probs, dim = -1).transpose(0, 1)
                
            #     label = batch["labels"][i:i+1]
            #     labels_for_metrics = label.clone()
            #     # Compute WER directly
            #     wer_val = compute_metrics(labels_for_metrics, pred, head_idx)
            #     lang_wers[head_idx].append(wer_val)
            ############################################################












            # # batched in gropby langids
            # #====================================
            # for lang_id in unique_lang_ids:
                
            #     #get indices for this lang_id
            #     indices = (lang_ids == lang_id).nonzero(as_tuple=True)[0]
                
            #     if is_main_process():
            #         print(indices)
                
            #     if len(indices) == 0:
            #         continue
                
            #     # Slice batch for this lang
            #     lang_shared_out = shared_output[indices]
            #     if is_main_process():
            #         print('the shape of the lang_shared_out', lang_shared_out.shape)
                
            #     lang_labels = batch["labels"][indices]
            #     if is_main_process():
            #         print('the shape of the lang_labels', lang_labels.shape)
            #     lang_input_lens = input_lengths[indices]
                
            #     if is_main_process():
            #         print('the length input length is', lang_input_lens)
                
            #     lang_label_lens = label_lengths[indices]
                
            #     if is_main_process():
            #         print('the lang_label_lens', lang_label_lens)
            #     head = multilingual_asr_model.module.ctc_heads[lang_id.item()]
                
            #     # Forward pass through head
            #     log_probs = F.log_softmax(head(lang_shared_out), dim=-1).transpose(0, 1)  # SInce it is batched
            #     # Loss
            #     loss = ctc_loss_fn(log_probs, lang_labels, lang_input_lens, lang_label_lens)
            #     total_loss += loss
                
            #     pred = torch.argmax(log_probs, dim = -1).transpose(0, 1)
                
                
            #     for i, idx in enumerate(indices):
            #         label = batch["labels"][idx:idx+1]
            #         pred_i = pred[i:i+1]
            #         wer_val = compute_metrics(label.clone(), pred_i, lang_id.item())
            #         lang_wers[lang_id.item()].append(wer_val)




# For inference 
# def infer(model, input_values, attention_mask, lang_id):
#     """
#     Perform inference for a single input (or batch with same language).
#     Args:
#         model: MultiHeadCTCModel
#         input_values: Tensor of shape [B, T]
#         attention_mask: Tensor of shape [B, T]
#         lang_id: int or Tensor of shape [B], the language index
#     Returns:
#         log_probs: Tensor of shape [B, T, V] - log probabilities from the appropriate head
#     """
#     with torch.no_grad():
#         outputs = model.wav2vec2(input_values, attention_mask=attention_mask)
#         hidden_states = model.dropout(outputs.last_hidden_state)

#         if isinstance(lang_id, int):
#             # Inference for a single language
#             head = model.heads[lang_id]
#             log_probs = F.log_softmax(head(hidden_states), dim=-1)  # [B, T, V]
#         else:
#             # Inference for batch with multiple languages
#             log_probs = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], model.heads[0].out_features, device=hidden_states.device)
#             for i in range(hidden_states.shape[0]):
#                 head = model.heads[lang_id[i].item()]
#                 log_probs[i] = F.log_softmax(head(hidden_states[i]), dim=-1)

#         return log_probs