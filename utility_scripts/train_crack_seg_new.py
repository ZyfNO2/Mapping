"""
Train YOLO model for crack segmentation - New Model: crack-seg
"""

from ultralytics import YOLO
import os


def train_crack_segmentation():
    """Train YOLO model on crack segmentation dataset"""
    
    # Load YOLO26n-seg model
    model = YOLO('yolo26n-seg.pt')
    
    print("="*60)
    print("Starting Crack Segmentation Training")
    print("Model name: crack-seg")
    print("="*60)
    
    # Train the model
    results = model.train(
        data='trainData/crack_segmentation/dataset.yaml',  # dataset YAML
        epochs=100,  # number of epochs
        imgsz=640,  # image size
        batch=16,  # batch size
        device='0',  # GPU device
        workers=8,  # number of workers
        patience=20,  # early stopping patience
        save=True,  # save best model
        project='crack-seg',  # project name - NEW MODEL NAME
        name='train',  # experiment name
        exist_ok=True,  # overwrite existing
        pretrained=True,  # use pretrained weights
        optimizer='AdamW',  # optimizer
        lr0=0.001,  # initial learning rate
        lrf=0.01,  # final learning rate
        momentum=0.937,  # SGD momentum
        weight_decay=0.0005,  # weight decay
        warmup_epochs=3.0,  # warmup epochs
        warmup_momentum=0.8,  # warmup momentum
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain
        dfl=1.5,  # dfl loss gain
        mask_ratio=4,  # mask downsample ratio
        overlap_mask=True,  # masks should overlap during training
        verbose=True,  # verbose output
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best model saved at: {results.best}")
    print("="*60)
    
    # Validate the model
    print("\nValidating model...")
    metrics = model.val()
    print(f"\nValidation metrics:")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    
    return model, results


def export_model(model):
    """Export trained model to various formats"""
    
    print("\n" + "="*60)
    print("Exporting model...")
    print("="*60)
    
    # Export to ONNX
    try:
        model.export(format='onnx', dynamic=True)
        print("✓ Exported to ONNX format")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
    
    # Export to TorchScript
    try:
        model.export(format='torchscript')
        print("✓ Exported to TorchScript format")
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
    
    # Export to TensorRT (if on GPU)
    try:
        model.export(format='engine', half=True)
        print("✓ Exported to TensorRT format")
    except Exception as e:
        print(f"✗ TensorRT export failed: {e}")


if __name__ == "__main__":
    # Check if dataset is prepared
    if not os.path.exists('trainData/crack_segmentation/dataset.yaml'):
        print("Error: Dataset not prepared yet!")
        print("Please run: python prepare_crack_seg_data_fast.py")
        exit(1)
    
    # Check if model exists
    if not os.path.exists('yolo26n-seg.pt'):
        print("Error: yolo26n-seg.pt not found!")
        print("Please download the model first.")
        exit(1)
    
    # Train model
    model, results = train_crack_segmentation()
    
    # Export model
    export_model(model)
    
    print("\n" + "="*60)
    print("All tasks completed!")
    print(f"Model: crack-seg")
    print(f"Results saved in: crack-seg/train/")
    print("="*60)
