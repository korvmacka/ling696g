from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, AutoFeatureExtractor

# Load Swedish Wav2Vec2 model
model_name = "KBLab/wav2vec2-large-voxrex-swedish"
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Load tokenizer separately (use a fallback tokenizer if missing)
try:
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
except OSError:
    print("No tokenizer found in model! Using a fallback tokenizer.")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-960h")

# Load feature extractor separately
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Combine into a processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Save model and processor
model.save_pretrained("/home/u5/shawnabirnbaum/wav2vec2_sv")
processor.save_pretrained("/home/u5/shawnabirnbaum/wav2vec2_sv")

print("Wav2Vec2 model and processor saved with proper feature extractor!")
