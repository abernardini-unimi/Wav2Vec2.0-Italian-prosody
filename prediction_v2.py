import torch
import torch.nn as nn
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2FeatureExtractor, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import librosa
import numpy as np
import sys
import os

# =============================================================================
# 1. DEFINIZIONE DELLA CLASSE DEL MODELLO (Deve essere identica al training)
# =============================================================================
class Wav2Vec2ForEmotionRegression(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # Head di regressione
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        
        # Fix per compatibilità
        self._tied_weights_keys = []
        self.post_init()

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        pooled_output = torch.mean(hidden_states, dim=1)
        
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        
        # Sigmoide per output 0-1
        preds = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = None # In inferenza non calcoliamo la loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=preds, # Qui restituisce già i valori tra 0 e 1
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# =============================================================================
# 2. FUNZIONE DI PREDIZIONE
# =============================================================================
def predict_emotion(audio_path, model_path="./wav2vec_output"):
    
    # Verifica dispositivo (CPU o GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Caricamento modello da '{model_path}' su {device}...")

    try:
        # Carica Configurazione e Feature Extractor
        config = AutoConfig.from_pretrained(model_path)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        
        # Carica il Modello
        model = Wav2Vec2ForEmotionRegression.from_pretrained(model_path, config=config)
        model.to(device)
        model.eval() # Imposta in modalità valutazione (spegne dropout, ecc.)
    except OSError:
        print(f"❌ Errore: Non trovo il modello nella cartella '{model_path}'.")
        print("   Assicurati di puntare alla cartella dove è stato salvato (es. ./wav2vec_output/checkpoint-500)")
        return

    print(f"🎤 Analisi file: {audio_path}")

    # Caricamento e Resampling Audio a 16kHz
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"❌ Errore nel caricamento del file audio: {e}")
        return

    # Preprocessing
    inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Inferenza
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.cpu().numpy()[0] # Estrae i 3 valori

    # Output Risultati
    arousal = predictions[0]
    valence = predictions[1]
    dominance = predictions[2]

    print("\n📊 RISULTATI PROSODIA:")
    print("-" * 30)
    print(f"⚡ Arousal (Attivazione): {arousal:.4f}  [{'Calmo' if arousal < 0.5 else 'Eccitato'}]")
    print(f"😊 Valence (Positività):  {valence:.4f}  [{'Negativo' if valence < 0.5 else 'Positivo'}]")
    print(f"👑 Dominance (Controllo): {dominance:.4f}  [{'Sottomesso' if dominance < 0.5 else 'Dominante'}]")
    print("-" * 30)

# =============================================================================
# 3. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    # Cambia questo percorso con il file che vuoi testare
    file_audio = "test_audio.wav" 
    
    # Se passi il file come argomento da terminale (es: python predict.py mio_file.wav)
    if len(sys.argv) > 1:
        file_audio = sys.argv[1]

    # Controlla se il file esiste
    if not os.path.exists(file_audio):
        print(f"⚠️ Attenzione: Il file '{file_audio}' non esiste. Inserisci un percorso valido.")
    else:
        # PUNTI AL MODELLO: Assicurati che questa cartella contenga 'config.json' e 'pytorch_model.bin'
        # Se il training si è interrotto, prova a cercare dentro ./wav2vec_output/checkpoint-XXX
        cartella_modello = "./wav2vec_output" 
        
        predict_emotion(file_audio, model_path=cartella_modello)