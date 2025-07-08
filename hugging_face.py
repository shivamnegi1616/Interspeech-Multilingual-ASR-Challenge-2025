from huggingface_hub import HfApi
import os
import re

api = HfApi(token=os.getenv("HF_TOKEN"))

model_folder_path = "/scratch/IITB/ai-at-ieor/23m1508/23m1508_backup/checkpoint_points/weights_810000_830000"


    
api.upload_folder(
    folder_path=model_folder_path,
    repo_id="ai-at-ieor/multilingual_asr_model",
    repo_type="model",
)

# Run this
# python /home/IITB/ai-at-ieor/23m1508/Shivam_23M1508/Interspeech/code/hugging_face.py