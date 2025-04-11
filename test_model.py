import os
import torch
from PIL import Image, ImageDraw
from transformers import ViTImageProcessor
from baseline_transformer import BaselinePredictor
import numpy as np

def load_model(model_path='baseline_predictor.pth'):
    # Initialize processor and model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = BaselinePredictor()
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, processor

def predict_baseline(model, processor, image_path):
    # Load and process image
    image = Image.open(image_path).convert('RGB')  # Ensure RGB format
    inputs = processor(images=image, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(inputs['pixel_values'])
    
    # Convert normalized coordinates back to image coordinates
    points = outputs[0].numpy()  # Get first batch item
    points[:, 0] = points[:, 0] * image.width
    points[:, 1] = points[:, 1] * image.height
    
    return points

def draw_baseline(image_path, points, output_path):
    # Load image
    image = Image.open(image_path).convert('RGB')  # Ensure RGB format
    draw = ImageDraw.Draw(image)
    
    # Draw baseline points
    for i in range(len(points) - 1):
        draw.line([tuple(points[i]), tuple(points[i+1])], fill='red', width=2)
    
    # Save annotated image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

def main(image_path=None):
    # Load model
    model, processor = load_model()
    
    if image_path is None:
        # Process all images in output directory
        output_dir = 'test_images'
        test_dir = 'tests'
        os.makedirs(test_dir, exist_ok=True)
        
        for img_name in os.listdir(output_dir):
            if img_name.endswith('.png') and not img_name.endswith('_annotated.jpg'):
                img_path = os.path.join(output_dir, img_name)
                output_path = os.path.join(test_dir, f'predicted_{img_name}')
                
                # Predict baseline
                points = predict_baseline(model, processor, img_path)
                
                # Draw and save prediction
                draw_baseline(img_path, points, output_path)
                print(f'Processed {img_name} -> {output_path}')
    else:
        # Process single image
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return
            
        output_path = os.path.join('tests', f'predicted_{os.path.basename(image_path)}')
        points = predict_baseline(model, processor, image_path)
        draw_baseline(image_path, points, output_path)
        print(f'Processed {image_path} -> {output_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test baseline prediction model')
    parser.add_argument('--image', type=str, help='Path to input image (optional)')
    args = parser.parse_args()
    
    main(args.image)
