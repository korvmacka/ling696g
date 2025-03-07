
# Introduction

My goal for the final project was to fine-tune a Wav2Vec model for Swedish speech recognition using Common Voice data. I had initially attempted to train a model using NeMo, but its pretraining was optimized for English phonetics, which produced a fun, but unintelligible Swenglish word salad. Instead of training a tokenizer from scratch, I switched to Wav2Vec which already has pretrained Swedish models available.

## Baseline Model
### Initial Model Setup
The first version of the model was based on the existing Swedish Wav2Vec model (KBLab’s wav2vec-large-voxrex), available on Hugging Face. The dataset was derived from Common Voice, focusing on the validated.tsv’s audio-transcription pairs. I filtered the dataset to around 20 hours of audio thereafter.

#### Steps
1. Extract audio-text pairs from `validated.tsv`, creating a subset of 20 hours:
`/home/u5/shawnabirnbaum/data/validated_20h.tsv`
2. Converted the dataset to Hugging Face format:
``` python
from datasets import Dataset
import pandas as pd

df = pd.read_csv("/home/u5/shawnabirnbaum/data/validated_20h.tsv", sep="\t", names=["path", "sentence"])
dataset = Dataset.from_pandas(df)
dataset.save_to_disk("/home/u5/shawnabirnbaum/data/preprocessed_swedish")
```
3. Stored the dataset at:
`/home/u5/shawnabirnbaum/data/preprocessed_swedish`

#### Initial training
The baseline model was trained on **20 hours of Swedish speech** using `train_wav2vec.py` inside a **Singularity container** on the University of Arizona HPC.

#### Training Setup

-   **Model:** `wav2vec2-large-voxrex`
-   **Dataset:** Common Voice Swedish (20 hours)
-   **Batch size:** 4
-   **Epochs:** 5
-   **Optimizer:** AdamW
-   **GPU:** Tesla P100 (via SLURM)
-   **Job Script:** `train_wav2vec.slurm`

**SLURM Submission Command**
`sbatch train_wav2vec.slurm`
**Training Output**
``` vbnet
Using device: Tesla P100-PCIE-16GB
Word Error Rate (WER): 0.9429
```
#### Initial Model Performance
The baseline WER was 94.29%, which lends itself to significant room for improvement. At the same time, qualitative transcription tests were promising, as the model was mostly accurate in recognizing full words; however, it struggled with out-of-vocabulary words and some tokenization problems.

## Tokenizer Fix & Retraining
### Identifying Tokenization Issues
After initial training, there were two key tokenization issues:
1. Incorrect special tokens: the tokenizer had `<pad>`, `<unk>`, and `<s>` tokens that were interfering with clean transcriptions.
2. No lowercase letters: the vocabulary lacked `a-z, å, ä, ö`, which led to errors.
### Fixing the Tokenizer
To fix this, I manually added lowercase characters and special tokens to the tokenizer:
``` bash
singularity exec --nv /contrib/hammond/nemo2.sif python3 -c "
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained('/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned')
processor.tokenizer.add_tokens(list('abcdefghijklmnopqrstuvwxyzåäö'))
processor.save_pretrained('/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned')
"
```
### Tokenizer Output (post-fix)
``` bash
{"'": 45, '0': 44, '1': 39, ..., 'Ö': 25, '[PAD]': 46, 'a': 47, ..., 'ö': 75}
```
This confirmed that lowercase letters were successfully added to the tokenizer.

### Retraining with Updated Tokenizer
**Training Output**
``` vbnet
Using device: Tesla P100-PCIE-16GB
Word Error Rate (WER): 0.9389
```
### Model Performance After Tokenizer Fix
There was a slight WER improvement (roughly .40%). However, some transcriptions became worse, possibly due to misalignment of token embeddings.

**Before**
``` bash
Transcription: nu ska du göra mig en tjänst klement
```
**After**
``` bash
Transcription: nu ska du göra mig en tjänst kleament
```
This probably suggests that retraining on a larger dataset might be necessary.

### Next Steps

1.  **Re-evaluate dataset preprocessing**:
    
    -   Ensure `validated_20h.tsv` is correctly preprocessed.
    -   Double-check that all training audio files exist.
2.  **Re-train model with cleaned dataset**:
    
    -   Use `/home/u5/shawnabirnbaum/data/preprocessed_swedish`
    -   Confirm training paths match dataset paths.
3.  **Evaluate final WER & transcription quality**:
    
    -   Compare model versions **before and after tokenizer fixes**.
    -   Analyze errors in `eval_wav2vec.py`.

# **Dataset Optimization & Restart After Storage Loss**

## **HPC Storage Issues & Restart**

After the initial training run, I **ran out of storage** on my **HPC home directory**. In attempting to **clear space**, I accidentally deleted **critical training files** that I was unable to recover.

This forced me to:

-   **Reprocess the dataset from scratch.**
-   **Rename and modify scripts** to reflect the restart.

### **Updated Script Names**

-   `train1.py` (Training script)
-   `train1.slurm` (SLURM job submission)
-   `eval1.py` (WER evaluation)
-   `infer1.py` (Inference/testing)

----------

## **Final Dataset Conversion**

With the cleaned TSV, I **converted it to Hugging Face format**:


``` python
import pandas as pd
from datasets import Dataset

df = pd.read_csv("/home/u5/shawnabirnbaum/data/validated_20h.tsv", sep="\t")
dataset = Dataset.from_pandas(df)
dataset.save_to_disk("/home/u5/shawnabirnbaum/data/preprocessed_swedish")

print("Dataset cleaned & saved for training (only audio & text)!")` 
```

**Final dataset path:**  
`/home/u5/shawnabirnbaum/data/preprocessed_swedish`

----------

## **Final Training Configuration**

### **Reverting to the Original Fine-Tuning Setup**

-   **Pretrained Model:** `KBLab/wav2vec2-large-voxrex`
-   **Dataset:** Common Voice Swedish (20 hours)
-   **Batch Size:** 8
-   **Epochs:** 2
-   **Optimizer:** AdamW
-   **Gradient Accumulation:** 4
-   **GPU:** Tesla P100
-   **Gradient Checkpointing:** Enabled

### **SLURM Job Submission**

`sbatch train1.slurm`

Training took **~88 minutes**.

## **Final Model Performance**

### **Final WER Calculation**

`Using device: Tesla P100-PCIE-16GB
Word Error Rate (WER): 0.9752` 

The **final WER was 97.52%**, which is **worse than the baseline model.**
### **Explanation for High WER**

Upon investigation, the issue was traced to a **missing `added_tokens.json` file**, which previously contained manually added lowercase letters (`a-z`, `å, ä, ö`). Because this file **was not carried over**, the tokenizer **did not recognize lowercase letters during inference**, leading to excessive `<pad>` tokens in the transcriptions. At the time of this writing, this discovery was only made 30 minutes before the final project submission deadline, so there was no time to make the proper corrections in the script.

-   **Potential Fix:** Restoring `added_tokens.json` and re-running inference would likely improve results without retraining.

## Thoughts 

The project successfully ran, as much as it could be considered "successful" for training a **Swedish ASR model**. But due to an oversight on my part, critical training files were missed in addition to a misstep in implementing the newer `added_tokens.json` - the variation I wanted to try in attempting to improve the WER score.

I will submit any and all files I manage to find related to this project.