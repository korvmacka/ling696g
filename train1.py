import os
import torch
import torchaudio
from datasets import load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

#  Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU detected. Running on CPU.")

#  Load dataset
dataset_path = "/home/u5/shawnabirnbaum/data/preprocessed_swedish"
dataset = load_from_disk(dataset_path)

#  Ensure dataset is split
if "train" not in dataset or "test" not in dataset:
    dataset = dataset.train_test_split(test_size=0.1)

#  Load Wav2Vec2 model and processor
model_name = "/home/u5/shawnabirnbaum/wav2vec2_sv"
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)
#  Enable Gradient Checkpointing to save memory
model.gradient_checkpointing_enable()


#  Fix tokenizer padding issue
processor.tokenizer.pad_token = "[PAD]"

#  Dynamically update vocab size
model.config.vocab_size = len(processor.tokenizer)
model.tie_weights()
print(f" Updated model vocab size: {model.config.vocab_size}")

#  Preprocessing function
def preprocess_function(batch):
    audio, _ = torchaudio.load(batch["path"])
    batch["input_values"] = processor(audio.squeeze().numpy(), sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["sentence"], padding="longest", return_tensors="pt").input_ids[0]
    return batch

dataset = dataset.map(preprocess_function, remove_columns=["path", "sentence"])

#  Custom Data Collator (Manual Padding)
def data_collator(features):
    input_values = [torch.tensor(f["input_values"]) for f in features]
    labels = [torch.tensor(f["labels"]) for f in features]

    input_values = pad_sequence(input_values, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  

    return {"input_values": input_values, "labels": labels}

#  Training arguments
training_args = TrainingArguments(
    output_dir="/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned",
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    num_train_epochs=2,
    logging_dir="/home/u5/shawnabirnbaum/logs",
    fp16=True,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_accumulation_steps=4  # ✅ Comma added!
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator  # ✅ Keep this, but remove dataloaders!
)

trainer.train()
model.save_pretrained("/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned")
processor.save_pretrained("/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned")
