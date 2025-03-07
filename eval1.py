import os
import torch
import torchaudio
import jiwer
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ✅ Define dataset & model paths
dataset_path = "/home/u5/shawnabirnbaum/data/validated_20h.tsv"
audio_dir = "/home/u5/shawnabirnbaum/data/wavs/"  # ✅ Confirm this is the correct directory!
model_path = "/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned"

# ✅ Load model & processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_path)

# ✅ Read dataset
df = pd.read_csv(dataset_path, sep="\t")

# ✅ WER storage
predictions = []
references = []

# ✅ Process each file **one at a time**
print("Running inference...")
for index, row in df.iterrows():
    try:
        # ✅ Prepend full directory path
        audio_path = os.path.join(audio_dir, row["path"])

        if not os.path.exists(audio_path):
            print(f"Skipping missing file: {audio_path}")
            continue  # ✅ Skip missing files instead of crashing

        # ✅ Load audio
        waveform, _ = torchaudio.load(audio_path)
        input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(predicted_ids)[0]

        # ✅ Store results for WER
        predictions.append(prediction)
        references.append(row["sentence"])

        # ✅ Print progress every 100 files
        if index % 100 == 0:
            print(f"Processed {index}/{len(df)} files...")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

# ✅ Compute final WER
wer = jiwer.wer(references, predictions)
print(f"\nFinal Word Error Rate (WER): {wer:.4f}")

