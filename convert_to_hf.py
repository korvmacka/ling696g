import pandas as pd
from datasets import Dataset

# Load TSV
df = pd.read_csv("/home/u5/shawnabirnbaum/data/validated_20h.tsv", sep="\t")

# Select only 'path' and 'sentence' (text transcription)
df = df[["path", "sentence"]]

# Convert paths to absolute file paths
df["path"] = df["path"].apply(lambda x: f"/home/u5/shawnabirnbaum/data/wavs/{x}")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset.save_to_disk("/home/u5/shawnabirnbaum/data/preprocessed_swedish")

print(" Dataset cleaned & saved for training (only audio & text)!")

