import json
import csv
import re
import os
import torch
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
import torch.nn as nn
from torch.utils.data import DataLoader
# AdamW is best optimizer
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer


import warnings
warnings.filterwarnings("ignore")


# Run this script
# python /home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/wav2vec2_xls-r_approach_1.py


# -------------------------------------------------------
# Currently in this notebook i am trianing the full model
# -------------------------------------------------------

# # only modify this config dictionary
# config = {
  
#             "CSV_DATA_PATH" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/data/scratch_balanced_shuffled_data.csv",
            
#             "VOCAB_PATH" : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/multilingual_vocab.json",
            
#             "BASE_MODEL_ID" : "facebook/wav2vec2-xls-r-2b",
            
#             "start_data" : 200000,
#             "end_data" :   300000,
#             'device' : 0,
#             'BATCH_SIZE' : 8,
#             'EPOCHS' : 2,
#             'LR' : 1e-5,
            
#             # First 10 layers of the transformer is frozen
#             'Number_of_first_layers_freeze_transofrmer' : 10,
            
#             'num_warmup_steps' : 500,
            
#             'loss_csv_saving_path'  : "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_loss_files/loss_2_3_lakh",
            
#             'loss_csv_name' : "loss_2_3_lakh.csv",
             
#             'prev_checkpoint_dir' : '/scratch/IITB/ai-at-ieor/23m1508/23m1508_backup/checkpoint_points/weights_0_2_lakh/multilingual_asr_model_0_2_lakh.pt',
            
#             # saving the tensors in the scratch
#             'new_checkpoint_dir' : "/scratch/IITB/ai-at-ieor/23m1508/23m1508_backup/checkpoint_points/weights_2_3_lakh",
            
#             'model_name' : "multilingual_asr_model_2_3_lakh",
            
#             'load_from_prev' : True,
#             'resume_training' : False
          
#           }


def get_latest_checkpoint(base_dir, prefix = "weights_", model_prefix = "multilingual_asr_model_"):
    
    """
    Finds the latest checkpoint directory based on numeric ranges in the names.
    Returns the last_end_index and full path of the checkpoint.
    """
    
    if not os.path.exists(base_dir):
        return 0, None

    checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    
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
    
    return max_end, latest_ckpt_path


# Declare globals outside
new_start = None
new_end = None

def generate_config(base_dir, csv_path, vocab_path, model_id, loss_dir_base, chunk_size, batch_size,  epochs, lr, freeze_layers):
    
    
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
                "VOCAB_PATH": vocab_path,
                "BASE_MODEL_ID": model_id,
                "start_data": new_start,
                "end_data": new_end,
                "device": 0,
                "BATCH_SIZE": batch_size,
                "EPOCHS": epochs,
                "LR": lr,
                "Number_of_first_layers_freeze_transofrmer": freeze_layers,
                "num_warmup_steps": 500,
                "loss_csv_saving_path": loss_dir_base,
                "loss_csv_name": f"loss_{new_start}_{new_end}.csv",
                "prev_checkpoint_dir": latest_checkpoint if latest_checkpoint else "",
                "new_checkpoint_dir": new_checkpoint_path,
                "model_name": f"multilingual_asr_model_{new_start}_{new_end}",
                "load_from_prev": latest_checkpoint is not None,
                "resume_training": False
            }

    print('The new start: ', new_start, 'The new start: ', new_end)
    return config



config = generate_config(
    
                base_dir = "/scratch/IITB/ai-at-ieor/23m1508/23m1508_backup/checkpoint_points",
                csv_path = "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/data/scratch_balanced_shuffled_data.csv",
                vocab_path = "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_files/multilingual_vocab.json",
                model_id = "facebook/wav2vec2-xls-r-2b",
                loss_dir_base = "/home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/model_code/model_loss_files", 
                chunk_size = 25000, 
                batch_size = 12, 
                epochs = 1,
                lr = 1e-5, 
                freeze_layers = 10
            )


print('The configration is :', config)

device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")

print(device)

final_data = pd.read_csv(config['CSV_DATA_PATH'])


final_data = final_data.loc[config['start_data'] : config['end_data'], :]


def lowercase_english_letters(text):
    return ''.join([c.lower() if c.isascii() and c.isalpha() else c for c in text])

final_data['Text'] = final_data['Text'].apply(lowercase_english_letters)


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


# Building the tokenizer

tokenizer = Wav2Vec2CTCTokenizer(config['VOCAB_PATH'],
                                 bos_token = "<s>",
                                 eos_token = "</s>",
                                 unk_token = "<unk>", 
                                 pad_token = "<pad>", 
                                 word_delimiter_token = "|")


feature_extractor = Wav2Vec2FeatureExtractor(feature_size = 1, 
                                             sampling_rate = 16000, 
                                             padding_value = 0.0, 
                                             do_normalize = True, 
                                             return_attention_mask = True)


processor = Wav2Vec2Processor(feature_extractor = feature_extractor, 
                              tokenizer = tokenizer)


# Data class

class SpeechDataset(Dataset):
    def __init__(self, df, processor, transforms=None):
        self.df = df.reset_index(drop=True)  # ensure index is clean
        self.processor = processor
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sample_rate = torchaudio.load(row['Path'])
        text = row['Text']

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        waveform = waveform.squeeze()

        # Optional audio transforms
        if self.transforms:
            for transform in self.transforms:
                waveform = transform(waveform)

        # Process audio
        input_values = self.processor.feature_extractor(
                                                        waveform, sampling_rate=16000, return_tensors="pt"
                                                    )["input_values"].squeeze(0)

        # Process labels (text)
        labels = self.processor.tokenizer(
                                            text, return_tensors="pt", truncation=True
                                        )["input_ids"].squeeze(0)

        return {
                    "input_values": input_values,
                    "labels": labels
                }


speech_dataset = SpeechDataset(final_data, processor, transforms = None)

print("The length of the speech dataset", len(speech_dataset))


def collate_function(batch, processor, 
                     padding = True, 
                     max_length = None, 
                     max_length_labels = None, 
                     pad_to_multiple_of = None, 
                     pad_to_multiple_of_labels = None):
  
    # Extract input values and labels from each sample in the batch
    b_X = [{"input_values": sample["input_values"]} for sample in batch]
    b_Y = [{"input_ids": sample["labels"]} for sample in batch]

    # Pad the audio inputs the same length
    features = processor.feature_extractor.pad(
                                                b_X,
                                                padding = padding,
                                                max_length = max_length,
                                                pad_to_multiple_of = pad_to_multiple_of,
                                                return_tensors = "pt"
                                              )

    # Pad the labels
    batchY = processor.tokenizer.pad(
                                        b_Y,
                                        padding = padding,
                                        max_length = max_length_labels,
                                        pad_to_multiple_of = pad_to_multiple_of_labels,
                                        return_tensors = "pt"
                                    )

    # Replace padding tokens in labels with -100, so they are ignored during loss calculation
    labels = batchY["input_ids"].masked_fill(batchY.attention_mask.ne(1), -100)

    # Add the padded labels back into the features dictionary
    features["labels"] = labels

    # Return the features, which now include both input values and labels
    return features
  

collate_fn = partial(collate_function, 
                     processor = processor, 
                     padding = True, 
                     max_length = None, 
                     max_length_labels = None, 
                     pad_to_multiple_of = None, 
                     pad_to_multiple_of_labels = None)


batch_size = config['BATCH_SIZE']
epochs = config['EPOCHS']
lr = config['LR']

print('The batch size is :', batch_size)

train_dataloader = DataLoader(speech_dataset, 
                              batch_size = batch_size, 
                              shuffle = True, 
                              collate_fn = collate_fn,
                              pin_memory = True)


# Loaidng the pretrained model
model = Wav2Vec2Model.from_pretrained(config['BASE_MODEL_ID'])


# Freeze feature extractor (same as model.freeze_feature_encoder())
for param in model.feature_extractor.parameters():
    param.requires_grad = False


# Freeze feature projection (optional)
for param in model.feature_projection.parameters():
    param.requires_grad = False

# Freeze all transformer encoder layers
# for param in model.encoder.parameters():
#     param.requires_grad = False
    

# Freeze first transformer layers (out of 48 for base model)
if config['Number_of_first_layers_freeze_transofrmer'] is not None:
    
    print('The number of layers to freeze is :', config['Number_of_first_layers_freeze_transofrmer'])
    
    for i in range(config['Number_of_first_layers_freeze_transofrmer']):
        for param in model.encoder.layers[i].parameters():
            param.requires_grad = False
        
        
# Freeze all parameters in the model    
# for param in model.parameters():
#     param.requires_grad = False
    
# Optional: Print which layers are frozen
# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad = {param.requires_grad}")



class Projector(nn.Module):
    def __init__(self, model, projection_dim = 5000):
        super().__init__()
        self.wav2vec2 = model
        self.projection = nn.Linear(1920, projection_dim)

    def forward(self, input_values, attention_mask = None):
        outputs = self.wav2vec2(input_values, attention_mask = attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, time, hidden]
        projected = self.projection(hidden_states)  # [batch, time, 5000]
        return projected


# Custom CTC model
class CustomWav2Vec2CTC(nn.Module):
    def __init__(self, model, vocab_size, projection_dim = 5000):
        super().__init__()

        self.projector = Projector(model, projection_dim = projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(projection_dim, vocab_size)

    def forward(self, input_values, attention_mask = None):
        hidden_states = self.projector(input_values, attention_mask)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits
      


# Path to your JSON file
file_path = config['VOCAB_PATH']

# Open and load JSON data
with open(file_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)
    

vocab_size = len(vocab)  # your vocab size here
print('The length of the vocab is :', vocab_size)


# This is complete model
multilingual_asr_model = CustomWav2Vec2CTC(model, vocab_size = vocab_size)

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


scaler = torch.amp.GradScaler(device)

multilingual_asr_model = multilingual_asr_model.train()


ctc_loss_fn = nn.CTCLoss(blank = processor.tokenizer.pad_token_id, zero_infinity = True, reduction = 'mean')


# def compute_metrics(labels, preds):
    
#     # preds = torch.argmax(preds, axis=-1)

#     labels[labels == -100] = processor.tokenizer.pad_token_id

#     # print('The shape of the preds is', preds)
#     # print('The shape of the label is', labels)
#     pred_str = processor.batch_decode(preds)
    
#     # print('The pred str is', pred_str)
#     label_str = processor.batch_decode(labels, group_tokens = False)
#     # print('The label str is', label_str)
#     return wer(label_str, pred_str)
  

def compute_metrics(labels, preds):
    
    # preds = torch.argmax(preds, axis=-1)

    labels[labels == -100] = processor.tokenizer.pad_token_id

    # print('The shape of the preds is', preds)
    # print('The shape of the label is', labels)
    pred_str = processor.batch_decode(preds)
    
    # print('The pred str is', pred_str)
    label_str = processor.batch_decode(labels, group_tokens = False)
    
    # Filter out pairs where reference (label) is an empty string
    filtered_preds = []
    filtered_labels = []

    for ref, hyp in zip(label_str, pred_str):
        if ref.strip() != "":
            filtered_labels.append(ref)
            filtered_preds.append(hyp)
            
    if len(filtered_labels) == 0:
        print("Warning: All references are empty after filtering. Returning WER=1.0.")
        return {"wer": 1.0}  # or 0.0 depending on your preference
    
    # print('The filtered preds is', filtered_preds)
    # print('The filtered labels is', filtered_labels)
    
    return wer(filtered_labels, filtered_preds)


# Define CSV directory and file
csv_dir = config['loss_csv_saving_path']
csv_file_name = config['loss_csv_name']
csv_file_path = os.path.join(csv_dir, csv_file_name)

# Ensure the directory exists
os.makedirs(csv_dir, exist_ok=True)

# If the CSV file doesn't exist, create it with headers
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["step", "epoch", "loss", "wer"])
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
    
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu', weights_only = False)
    
    multilingual_asr_model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    
    curr_best_loss = checkpoint.get('best_loss', curr_best_loss)
    
    print('Loaded form the previous model: ')
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Trained till to epoch {start_epoch} with best loss {curr_best_loss:.4f}")
    

# Resume logic and save in new check dir since it is new chunk of the data
if resume_training and os.path.exists(new_checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    
    multilingual_asr_model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    
    curr_best_loss = checkpoint.get('best_loss', curr_best_loss)
    
    print(f"Resumed training from epoch {start_epoch} with best loss {curr_best_loss:.4f}")
    
    
multilingual_asr_model = multilingual_asr_model.to(device)


# Move optimizer state to the correct device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)
            
            
            


curr_best_loss = 1e9

for n in tqdm(range(start_epoch, start_epoch + 1)):
    
    print('The epoch start epoch is :', n)
    
    losses = []
    wers = []

    total_number_batch = len(train_dataloader)
    
    for step, batch in enumerate(tqdm(train_dataloader)):

        optimizer.zero_grad()
        batch = {k: v.to(device, non_blocking = True) for k, v in batch.items()}

        # print(torch.isnan(batch["input_values"]).any())
        # print(torch.isinf(batch["input_values"]).any())

        with torch.cuda.amp.autocast(enabled=False):
        # with torch.autocast("cuda"):
            logits = multilingual_asr_model(batch["input_values"], attention_mask = batch.get("attention_mask"))
            
            # print(torch.isnan(logits).any())  # If True, problem is in the model

        log_probs = F.log_softmax(logits, dim = -1).transpose(0, 1)

        input_lengths = torch.full(
                                    size = (log_probs.size(1),),  # batch size
                                    fill_value = log_probs.size(0),  # time dimension
                                    dtype = torch.long).to(device)


        labels = batch["labels"]
        
        target_lengths = (labels != -100).sum(dim = 1)

        flattened_targets = labels[labels != -100]

        with torch.cuda.amp.autocast(enabled = False):
            loss = ctc_loss_fn(
                                log_probs.float(),         # ensure float32
                                flattened_targets,
                                input_lengths,
                                target_lengths
                                )
        
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        lr_scheduler.step()
        scaler.update()

        # WER computation
        with torch.no_grad():
            preds = torch.argmax(log_probs, dim = -1).transpose(0, 1)  # back to (batch, time)
            
            labels_for_metrics = labels.clone()
            
            metrics = compute_metrics(labels_for_metrics, preds)
            wers.append(metrics)


        with open(csv_file_path, mode = 'a', newline = '') as file:
            writer = csv.DictWriter(file, fieldnames=["step", "epoch", "loss", "wer"])
            writer.writerow({
                                "step": step,
                                "epoch": n + 1,
                                "loss": loss.item(),
                                "wer": metrics
                            })
            
        # Explicit memory cleanup
        del batch, logits, log_probs, input_lengths, labels, target_lengths, flattened_targets, loss, preds
        torch.cuda.empty_cache()

    result = {"loss": np.mean(losses), "wer": np.mean(wers)}    
    
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

print(f'Training completed for the {new_start} to {new_end} chunk of data')



if new_end >= 1300000:  # Replace with your actual full data limit
    with open("done.flag", "w") as f:
        f.write("Training Completed\n")
        

print("Training completed for the chunk of data from {} to {}".format(new_start, new_end))
print("==" * 50)
