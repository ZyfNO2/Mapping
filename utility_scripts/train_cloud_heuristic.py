"""
Train YOLO model using Ultralytics Cloud API - Heuristic Elephant Project
"""
import os

def main():
    # Set API key
    os.environ['ULTRALYTICS_API_KEY'] = 'ul_06f17f6f1300f867af2edf68230f9fe258d21951'

    # Import ultralytics after setting API key
    from ultralytics import YOLO

    print("="*60)
    print("Starting Cloud Training - Heuristic Elephant Project")
    print("="*60)
    print("Model: YOLO26n-seg")
    print("Dataset: Cracknet Images")
    print("Project: ad-astra/heuristic-elephant")
    print("Epochs: 20")
    print("="*60)

    # Load model from cloud
    model = YOLO('ul://ultralytics/yolo26/yolo26n-seg')

    # Train with cloud dataset - New configuration
    results = model.train(
        data='ul://ad-astra/datasets/cracknet-images',
        epochs=20,
        batch=16,
        imgsz=640,
        project='ad-astra/heuristic-elephant',
        name='train',
        device='0',
        workers=0,  # Disable multiprocessing for Windows
        verbose=True,
    )

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Results saved to: ad-astra/heuristic-elephant/train/")
    print("="*60)

if __name__ == '__main__':
    main()
