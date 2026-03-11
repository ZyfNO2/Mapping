"""
Train YOLO model using Ultralytics Cloud API
"""
import os

def main():
    # Set API key
    os.environ['ULTRALYTICS_API_KEY'] = 'ul_06f17f6f1300f867af2edf68230f9fe258d21951'

    # Import ultralytics after setting API key
    from ultralytics import YOLO

    print("Starting cloud training with Ultralytics API...")
    print("="*60)

    # Load model from cloud
    model = YOLO('ul://ultralytics/yolo26/yolo26n-seg')

    # Train with cloud dataset
    results = model.train(
        data='ul://ad-astra/datasets/cracknet-images',
        epochs=100,
        batch=16,
        imgsz=640,
        project='ad-astra/example-project',
        name='crack-seg-cloud',
        device='0',
        workers=0,  # Disable multiprocessing for Windows
    )

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Results: {results}")

if __name__ == '__main__':
    main()
