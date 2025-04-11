import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

class BaselineDataset(Dataset):
    def __init__(self, image_dir, xml_dir, processor):
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.processor = processor
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and not f.endswith('_annotated.jpg')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        
        # Load corresponding XML
        xml_name = img_name.replace('.jpg', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract baseline points
        baseline = root.find('.//{http://www.loc.gov/standards/alto/ns-v4#}Baseline')
        points_str = baseline.get('POINTS')
        points = [tuple(map(float, p.split(','))) for p in points_str.split()]
        
        # Convert points to normalized coordinates (0-1)
        points = np.array(points)
        points[:, 0] = points[:, 0] / image.width
        points[:, 1] = points[:, 1] / image.height
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'baseline_points': torch.FloatTensor(points)
        }

class BaselinePredictor(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(pretrained_model_name)
        self.regression_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 20)  # 10 points * 2 coordinates
        )
        
    def forward(self, pixel_values):
        # Get features from ViT
        outputs = self.vit(pixel_values, output_hidden_states=True)
        features = outputs.hidden_states[-1][:, 0, :]  # Use [CLS] token
        
        # Predict baseline points
        points = self.regression_head(features)
        return points.view(-1, 10, 2)  # Reshape to (batch_size, 10, 2)

def train_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            pixel_values = batch['pixel_values'].to(device)
            baseline_points = batch['baseline_points'].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = criterion(outputs, baseline_points)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                baseline_points = batch['baseline_points'].to(device)
                
                outputs = model(pixel_values)
                loss = criterion(outputs, baseline_points)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')

def main():
    # Initialize processor and model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = BaselinePredictor()
    
    # Create dataset and dataloaders
    dataset = BaselineDataset('output', 'output', processor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Train model
    train_model(model, train_loader, val_loader)
    
    # Save model
    torch.save(model.state_dict(), 'baseline_predictor.pth')

if __name__ == '__main__':
    main() 