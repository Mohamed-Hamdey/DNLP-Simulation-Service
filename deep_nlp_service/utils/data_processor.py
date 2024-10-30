from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self, max_label_length: int = 10):
        self.max_label_length = max_label_length
        self.label_encoder = LabelEncoder()

    def prepare_data(self, data: List[Dict]) -> Tuple[List[str], List[List[float]]]:
        texts = [item['input'] for item in data]
        outputs = [self._flatten_dict(item['output']) for item in data]

        numerical_outputs = [
            self._pad_encoded_output(self.label_encoder.fit_transform([str(v)])[0] for v in output.values())

            for output in outputs
        ]

        return texts, numerical_outputs

    def _pad_encoded_output(self, encoded_output: List[float]) -> List[float]:
        # Pad or truncate to ensure consistent length
        return encoded_output[:self.max_label_length] + [0] * (self.max_label_length - len(encoded_output))

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)