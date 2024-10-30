from trainer import NLPService
from deep_nlp_service.config import ModelConfig


def main():
    # Create configuration
    config = ModelConfig(
        max_length=128,
        embedding_dim=768,
        hidden_dim=256,
        n_layers=2,
        dropout=0.3,
        num_heads=8,
        batch_size=32,
        learning_rate=0.001,
        epochs=10,
        model_path="best_model.pt"
    )
    
    # Initialize service without wandb
    service = NLPService(config=config)
    
    # Load your JSON data
    train_data = service.load_data('simulation_dataset.json')
    
    try:
        # Train the model
        print("Starting training...")
        service.train(train_data)
        
        # Save the model only after successful training
        print("Training completed. Saving model...")
        service.save_model('trained_model.pt')
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")

    # # Test a prediction
    # text = "Example text for prediction"
    # prediction = service.predict(text)
    # print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()