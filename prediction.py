import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf  # <--- NUOVO IMPORT NECESSARIO
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from src.models import Wav2Vec2ForSpeechClassification

model_name_or_path = "model/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate

# for wav2vec
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
model.eval() # <--- IMPORTANTE: Modalità valutazione (disattiva dropout)

def speech_file_to_array_fn(path, target_sampling_rate):
    # MODIFICA: Usiamo soundfile invece di torchaudio.load per stabilità su Windows
    speech_array, src_sampling_rate = sf.read(path)
    
    # Convertiamo da Numpy a Tensor Float
    speech_array = torch.from_numpy(speech_array).float()
    
    # Gestione dimensioni: Soundfile è (Time, Channels), Torch vuole (Channels, Time)
    if speech_array.ndim == 1:
        speech_array = speech_array.unsqueeze(0) # Diventa (1, Time)
    else:
        speech_array = speech_array.t() # Diventa (Channels, Time)
        
    # Se stereo, facciamo la media per avere mono (Wav2Vec vuole mono)
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

    # Ricampionamento se necessario
    if src_sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(src_sampling_rate, target_sampling_rate)
        speech_array = resampler(speech_array)
    
    # Ritorniamo numpy array spremuto (squeezed) come si aspetta il feature_extractor
    speech = speech_array.squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    
    # Padding e conversione in tensori
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    
    # Spostiamo tutto sul device corretto (GPU o CPU)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    return outputs

# ESECUZIONE
path = "1612871026347.wav"

try:
    start_time = time.time()
    outputs = predict(path, sampling_rate)
    end_time = time.time()
    print(end_time - start_time)
    # Ordiniamo i risultati per punteggio (più alto prima) per leggibilità
    outputs.sort(key=lambda x: float(x["Score"].strip('%')), reverse=True)
    print("Risultati predizione:")
    for out in outputs:
        print(out)
except Exception as e:
    print(f"Errore durante la predizione: {e}")