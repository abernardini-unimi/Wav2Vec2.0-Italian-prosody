from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

import transformers
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor


@dataclass
class DataCollatorCTCWithPadding:
    # ... (docstring uguale)
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if len(label_features) > 0:
            # Controlla se il primo elemento è una lista (regressione multi-output)
            # oppure un singolo valore (classificazione o regressione singola)
            first = label_features[0]
            if isinstance(first, (list, tuple)):
                # Regressione multi-output: [arousal, valence, dominance]
                d_type = torch.float
            elif isinstance(first, float):
                d_type = torch.float
            else:
                d_type = torch.long

            batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch