"""
Train YOLO model for crack segmentation
"""

from ultralytics import YOLO
import os


def train_crack_segmentation():
    """Train YOLO model on crack segmentation dataset"""
    
    # Load YOLO26n-seg model
    model = YOLO('yolo26n-seg.pt')
    
    # Train the model
    results = model.train(
        data='trainData/crack_segmentation/dataset.yaml',  # dataset YAML
        epochs=100,  # number of epochs
        imgsz=640,  # image size
        batch=16,  # batch size
        device='0',  # GPU device (use 'cpu' for CPU training)
        workers=8,  # number of workers
        patience=20,  # early stopping patience
        save=True,  # save best model
        project='crack_seg_runs',  # project name
        name='yolo26n_crack',  # experiment name
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
        seg=1.0,  # segmentation loss gain
    )
    
    print("Training complete!")
    print(f"Best model saved at: {results.best}")
    
    # Validate the model
    metrics = model.val()
    print(f"\nValidation metrics:")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    
    return model, results


def export_model(model):
    """Export trained model to various formats"""
    
    # Export to ONNX
    model.export(format='onnx', dynamic=True)
    print("Model exported to ONNX format")
    
    # Export to TorchScript
    model.export(format='torchscript')
    print("Model exported to TorchScript format")
    
    # Export to TensorRT (if on GPU)
    try:
        model.export(format='engine', half=True)
        print("Model exported to TensorRT format")
    except Exception as e:
        print(f"TensorRT export failed: {e}")


if __name__ == "__main__":
    # Check if dataset is prepared
    if not os.path.exists('trainData/crack_segmentation/dataset.yaml'):
        print("Dataset not prepared yet!")
        print("Please run: python prepare_crack_seg_data.py")
        exit(1)
    
    # Train model
    model, results = train_crack_segmentation()
    
    # Export model
    print("\nExporting model...")
    export_model(model)
    
    print("\n" + "="*50)
    print("Training and export complete!")
    print(f"Results saved in: crack_seg_runs/yolo26n_crack/")
