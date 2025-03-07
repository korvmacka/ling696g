import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

#  Load model & processor
model_path = "/home/u5/shawnabirnbaum/wav2vec2_sv_finetuned"
model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()
processor = Wav2Vec2Processor.from_pretrained(model_path)

#  Run inference
def transcribe(audio_path):
    audio, _ = torchaudio.load(audio_path)
    input_values = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

#  Test on a sample
test_audio = "/home/u5/shawnabirnbaum/common_voice_sv-SE_41823407.wav"
transcription = transcribe(test_audio)
print(f" Transcription: {transcription}")
