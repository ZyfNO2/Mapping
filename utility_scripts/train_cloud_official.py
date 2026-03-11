"""
Train YOLO model using Ultralytics Cloud API - Official Configuration
Based on Ultralytics Hub interface
"""
import os

def main():
    # Set API key
    os.environ['ULTRALYTICS_API_KEY'] = 'ul_06f17f6f1300f867af2edf68230f9fe258d21951'

    # Import ultralytics after setting API key
    from ultralytics import YOLO

    print("="*60)
    print("Starting Cloud Training - Crack Segmentation")
    print("="*60)
    print("Model: YOLO26n-seg")
    print("Dataset: Cracknet Images (11.3k images, 1 class)")
    print("Project: ad-astra/example-project-5")
    print("="*60)

    # Load model from cloud
    model = YOLO('ul://ultralytics/yolo26/yolo26n-seg')

    # Train with cloud dataset - Official configuration
    results = model.train(
        data='ul://ad-astra/datasets/cracknet-images',
        epochs=100,
        batch=16,
        imgsz=640,
        project='ad-astra/example-project-5',
        name='crack-seg',
        device='0',
        workers=0,  # Disable multiprocessing for Windows
        verbose=True,
    )

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Results saved to: ad-astra/example-project-5/crack-seg/")
    print("="*60)

if __name__ == '__main__':
    main()
