import torch #type: ignore
import librosa #type: ignore
import sys

from src.models import Wav2Vec2ForSpeechClassification #type: ignore
from transformers import Wav2Vec2FeatureExtractor, AutoConfig #type: ignore

def predict_emotion(audio_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from '{model_path}' su {device}...")

    config = AutoConfig.from_pretrained(model_path)
    setattr(config, 'pooling_mode', 'mean')
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()

    print(f"Analysis file: {audio_path}")
    speech, sr = librosa.load(audio_path, sr=16000)

    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.cpu().numpy()[0]

    print("\nResults:")
    print("-" * 30)
    print(f"Arousal   : {predictions[0]:.4f}")
    print(f"Valence   : {predictions[1]:.4f}")
    print(f"Dominance : {predictions[2]:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    file_audio = "<your_audio_path>"
    if len(sys.argv) > 1:
        file_audio = sys.argv[1]

    cartella_modello = '<your_model_path>'
    predict_emotion(file_audio, model_path=cartella_modello)