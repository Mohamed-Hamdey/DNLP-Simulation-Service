from sklearn.preprocessing import LabelEncoder
import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer
import json
from typing import Dict, List, Optional, Tuple
from deep_nlp_service.models.dataset import ServiceDataset
from deep_nlp_service.models.bert_model import EnhancedDeepNLPModel
from deep_nlp_service.utils.data_processor import DataProcessor
from deep_nlp_service.utils.logger import setup_logger
from deep_nlp_service.config import ModelConfig

class NLPService:
    def __init__(self, config: Optional[ModelConfig] = None, use_wandb=False):
        self.config = config or ModelConfig()
        self.use_wandb = use_wandb
        self.device = torch.device(
            self.config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        self.data_processor = DataProcessor()
        self.logger = setup_logger(__name__)

    def load_data(self, json_path: str) -> List[Dict]:
        """Load and validate training data from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            for item in data:
                if 'input' not in item or 'output' not in item:
                    raise ValueError("Each data item must contain 'input' and 'output' keys")
            
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def save_model(self, path: str = None):
        """Save the trained model and configuration."""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        save_path = path or self.config.model_path
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'label_encoder': self.data_processor.label_encoder
        }
        torch.save(model_state, save_path)
        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, path: str = None):
        """Load a trained model and configuration."""
        load_path = path or self.config.model_path
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            
            # Initialize model with loaded config
            config_dict = checkpoint['config']
            self.config = ModelConfig(**config_dict)
            
            # Create and load model
            self.model = self._create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.data_processor.label_encoder = checkpoint['label_encoder']
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded from {load_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _create_model(self):
        """Create a new model instance based on configuration."""
        return EnhancedDeepNLPModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads
        )
    def _collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad input_ids and attention_mask
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            [ids.clone().detach() for ids in input_ids],  # Use clone().detach() instead of torch.tensor()
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [mask.clone().detach() for mask in attention_mask],  # Use clone().detach() instead of torch.tensor()
            batch_first=True,
            padding_value=0
        )
        
        # Stack labels without padding since they're already consistent
        labels_tensor = torch.stack([label.clone().detach() for label in labels])  # Use clone().detach() instead of torch.tensor()
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'labels': labels_tensor
        }


    def train(self, train_data: List[Dict], validation_data: Optional[List[Dict]] = None):
        """Train the model with the provided data."""
        if self.use_wandb:
            wandb.init(project="deep-nlp-service")

        # Prepare data
        train_texts, train_labels = self.data_processor.prepare_data(train_data)
        self.config.output_dim = len(train_labels[0])

        # Initialize model
        self.model = self._create_model().to(self.device)
        
        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        # DataLoader with collate function
        train_dataset = ServiceDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move batch to the device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                predictions = self.model(input_ids, attention_mask)
                loss = criterion(predictions, labels)
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            # Scheduler step
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            # Log training loss
            if self.use_wandb:
                wandb.log({'train_loss': avg_loss, 'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})
            
            self.logger.info(f"Epoch [{epoch+1}/{self.config.epochs}], Loss: {avg_loss:.4f}")
            
        self.logger.info("Training completed.")


    def predict(self, text: str) -> torch.Tensor:
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
        
        self.model.eval()
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(
                encoding['input_ids'],
                encoding['attention_mask']
            )
            
            return outputs.cpu()
        
class DataProcessor:
    def __init__(self, max_label_length=10):
        self.max_label_length = max_label_length
        self.label_encoder = LabelEncoder()

    def prepare_data(self, data: List[Dict]) -> Tuple[List[str], List[List[float]]]:
        texts = [item['input'] for item in data]
        outputs = [self._flatten_dict(item['output']) for item in data]
        
        numerical_outputs = []
        for output in outputs:
            # Encode the label values
            encoded_output = [
                float(self.label_encoder.fit_transform([str(v)])[0]) 
                for v in output.values()
            ]
            # Pad or truncate encoded_output to ensure consistent length
            padded_output = encoded_output[:self.max_label_length] + \
                            [0] * (self.max_label_length - len(encoded_output))
            numerical_outputs.append(padded_output)
        
        return texts, numerical_outputs

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
