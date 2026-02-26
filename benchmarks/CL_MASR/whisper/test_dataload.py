from datasets import load_dataset
from IPython.display import Audio
from huggingface_hub import hf_hub_download
import yaml

# ds = load_dataset("sarulab-speech/commonvoice22_sidon", "ia", split="train", streaming=True, trust_remote_code=True)

# lens = 0

# for i, sample in enumerate(ds):
#     audio = sample['flac']
#     lens += len(audio)

# print(lens/16000)

ds = load_dataset("sarulab-speech/commonvoice22_sidon", "ia", split="train", streaming=True, trust_remote_code=True)
sample = next(iter(ds))
print(sample.keys())
print(type(sample["flac"]), sample["flac"].keys() if isinstance(sample["flac"], dict) else None)
