import torch
import torch.nn as nn
from torch.nn import MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from src.modeling_outputs import SpeechClassifierOutput


class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gold, pred):
        # FIX: Forziamo i target a diventare Float prima di calcolare la media
        gold = gold.float()
        
        # Pred e Gold shape: (batch_size, num_labels)
        
        # Medie
        gold_mean = torch.mean(gold, dim=0)
        pred_mean = torch.mean(pred, dim=0)
        
        # Varianze
        gold_var = torch.var(gold, dim=0, unbiased=False)
        pred_var = torch.var(pred, dim=0, unbiased=False)
        
        # Covarianza
        cov = torch.mean((gold - gold_mean) * (pred - pred_mean), dim=0)
        
        # CCC formula
        ccc = (2 * cov) / (gold_var + pred_var + (gold_mean - pred_mean) ** 2)
        
        # Loss = 1 - CCC (per minimizzare)
        loss = 1.0 - ccc
        
        # Ritorniamo la media delle loss delle 3 dimensioni
        return torch.mean(loss)
    

class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        # Qui num_labels sarà 3
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
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        # Post-init weights (importante per inizializzare la head randomicamente)
        self.init_weights()
        self.loss_fct = CCCLoss()

    @property
    def all_tied_weights_keys(self):
        """
        Fix per compatibilità con versioni recenti di Transformers.
        Impedisce l'AttributeError durante from_pretrained.
        """
        return {}

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
        
        # Gestione sicura dell'output (sia dizionario che tupla)
        if isinstance(outputs, dict): # o BaseModelOutput
            hidden_states = outputs['last_hidden_state']
        else:
            hidden_states = outputs[0]

        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # FORZIAMO l'uso della CCC Loss per la regressione dimensionale
            # Assumiamo che labels abbia shape (batch, 3)
            loss = self.loss_fct(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        