import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lxml import etree
import cv2
import math

class SyntheticCurveGenerator:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        
    def generate_random_text(self, min_length=20, max_length=50):
        """Generate random text of varying length."""
        length = random.randint(min_length, max_length)
        return ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(length))
    
    def generate_curve(self, width, height, curve_type="random"):
        """Generate curve points for the baseline."""
        num_points = 10  # Reduced from 100 to 10 points
        x_points = np.linspace(0, width, num_points)
        
        if curve_type == "random":
            # Generate random curve
            amplitude = random.uniform(0.01, 0.1) * height
            frequency = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2 * math.pi)
            y_points = height/2 + amplitude * np.sin(frequency * x_points/width * 2 * math.pi + phase)
        else:
            # Generate straight line
            y_points = np.ones_like(x_points) * height/2
            
        # Round coordinates to 2 decimal places
        points = list(zip(np.round(x_points, 2), np.round(y_points, 2)))
        return points
    
    def draw_text_along_curve(self, text, curve_points, image_size=(800, 200), annotated=False):
        """Draw text along a curved baseline."""
        image = Image.new('RGB', image_size, 'white')
        draw = ImageDraw.Draw(image)
        
        # Calculate text line polygon
        text_height = 40  # Approximate text height
        top_points = [(x, y - text_height) for x, y in curve_points]
        bottom_points = [(x, y + text_height) for x, y in curve_points]
        text_polygon = top_points + bottom_points[::-1]  # Combine top and bottom points
        
        if annotated:
            # Draw text line polygon
            draw.polygon(text_polygon, outline='blue', width=1)
            
            # Draw baseline
            draw.line(curve_points, fill='red', width=1)
        
        # Calculate total text width and adjust spacing
        total_text_width = self.font.getlength(text)
        available_width = image_size[0] - 10  # Leave 5px margin on each side
        char_spacing = (available_width - total_text_width) / (len(text) - 1) if len(text) > 1 else 0
        
        # Draw text along curve
        x_offset = 5  # Start 5 pixels from left
        
        for char in text:
            # Find closest point on curve
            closest_point = min(curve_points, key=lambda p: abs(p[0] - x_offset))
            angle = math.atan2(curve_points[1][1] - curve_points[0][1], 
                             curve_points[1][0] - curve_points[0][0])
            
            # Draw character
            draw.text((x_offset, closest_point[1]), char, fill='black', font=self.font)
            x_offset += self.font.getlength(char) + char_spacing
            
        return image
    
    def create_alto_xml(self, text, curve_points, image_size, filename):
        """Create ALTO XML file with baseline information."""
        root = etree.Element("alto", xmlns="http://www.loc.gov/standards/alto/ns-v4#")
        layout = etree.SubElement(root, "Layout")
        page = etree.SubElement(layout, "Page", WIDTH=str(image_size[0]), HEIGHT=str(image_size[1]))
        print_space = etree.SubElement(page, "PrintSpace")
        
        # Create text block
        text_block = etree.SubElement(print_space, "TextBlock")
        
        # Add baseline information as polygon
        polygon_points = " ".join([f"{x},{y}" for x, y in curve_points])
        baseline = etree.SubElement(text_block, "Baseline", POINTS=polygon_points)
        
        # Add text line with polygon coordinates
        text_line = etree.SubElement(text_block, "TextLine")
        
        # Calculate polygon for text line (baseline + height)
        text_height = 40  # Approximate text height
        top_points = [(x, y - text_height) for x, y in curve_points]
        bottom_points = [(x, y + text_height) for x, y in curve_points]
        text_polygon = top_points + bottom_points[::-1]  # Combine top and bottom points
        
        text_line.set("POLYGON", " ".join([f"{x},{y}" for x, y in text_polygon]))
        
        # Add string element with content
        string = etree.SubElement(text_line, "String", CONTENT=text)
        
        # Write XML to file
        tree = etree.ElementTree(root)
        tree.write(os.path.join(self.output_dir, filename + ".xml"), 
                  pretty_print=True, 
                  xml_declaration=True, 
                  encoding="UTF-8")
    
    def generate_sample(self, num_samples=10):
        """Generate multiple samples of curved text."""
        for i in range(num_samples):
            text = self.generate_random_text()
            image_size = (800, 200)
            curve_points = self.generate_curve(image_size[0], image_size[1])
            
            # Generate original image
            image = self.draw_text_along_curve(text, curve_points, image_size, annotated=False)
            filename = f"sample_{i:03d}"
            image.save(os.path.join(self.output_dir, filename + ".jpg"))
            
            # Generate annotated image
            annotated_image = self.draw_text_along_curve(text, curve_points, image_size, annotated=True)
            annotated_image.save(os.path.join(self.output_dir, filename + "_annotated.jpg"))
            
            # Generate ALTO XML
            self.create_alto_xml(text, curve_points, image_size, filename)

if __name__ == "__main__":
    generator = SyntheticCurveGenerator()
    generator.generate_sample(1000) 