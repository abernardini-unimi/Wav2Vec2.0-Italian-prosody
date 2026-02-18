import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from src.modeling_outputs import SpeechClassifierOutput

class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gold, pred):
        # Assicuriamoci che siano float
        gold = gold.float()
        pred = pred.float()
        
        # Medie
        gold_mean = torch.mean(gold, dim=0)
        pred_mean = torch.mean(pred, dim=0)
        
        # Varianze
        gold_var = torch.var(gold, dim=0, unbiased=False)
        pred_var = torch.var(pred, dim=0, unbiased=False)
        
        # Covarianza
        cov = torch.mean((gold - gold_mean) * (pred - pred_mean), dim=0)
        
        # CCC formula
        numerator = 2 * cov
        denominator = gold_var + pred_var + (gold_mean - pred_mean) ** 2
        
        # Aggiungiamo un epsilon piccolissimo per evitare divisioni per zero (NaN)
        eps = 1e-8
        ccc = numerator / (denominator + eps)
        
        # Loss = 1 - CCC
        loss = 1.0 - ccc
        
        return torch.mean(loss)

class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()
        self.loss_fct = CCCLoss()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception("Pooling mode not supported")
        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 1. FIX LOGICA: Estrazione corretta senza sovrascrittura
        if isinstance(outputs, dict) or hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Pooling
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        
        # Logits grezzi
        logits = self.classifier(hidden_states)

        # 2. FIX MATEMATICO: Applicazione Sigmoide per output 0-1
        # Questo è fondamentale se i tuoi target sono normalizzati tra 0 e 1
        preds = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            # Calcoliamo la loss usando le PREDIZIONI (con sigmoide), non i logits
            loss = self.loss_fct(labels, preds)

        if not return_dict:
            output = (preds,) + outputs[2:] # Restituiamo preds, non logits
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=preds, # Restituiamo preds normalizzate
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )