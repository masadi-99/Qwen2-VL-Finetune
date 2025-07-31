#!/usr/bin/env python3
"""
Practical script to enhance ECG dataset for improved coordinate precision
Adds temporal context and coordinate verification training samples
"""

import json
import re
import os
import argparse
from typing import List, Dict
import random

def extract_coordinates_from_text(text: str) -> List[str]:
    """Extract coordinate values from <points> tags"""
    pattern = r'x\d+="(\d+(?:\.\d+)?)"'
    return re.findall(pattern, text)

def create_temporal_mapping_sample(original_sample: Dict, coords: List[str]) -> Dict:
    """Create a sample that teaches temporal mapping"""
    if len(coords) < 3:
        return None
    
    x1, x2, x3 = coords[0], coords[1], coords[2]
    
    return {
        "image": original_sample["image"],
        "conversations": [
            {
                "from": "human",
                "value": f"<image>Convert these ECG coordinates to milliseconds. Remember: each pixel = 2ms of time. Coordinates: x1={x1}, x2={x2}, x3={x3}"
            },
            {
                "from": "gpt",
                "value": f"Converting to milliseconds: x1={x1} pixels = {int(x1)*2}ms, x2={x2} pixels = {int(x2)*2}ms, x3={x3} pixels = {int(x3)*2}ms. Timeline: start at {int(x1)*2}ms, peak at {int(x2)*2}ms, end at {int(x3)*2}ms."
            }
        ]
    }

def create_coordinate_verification_sample(original_sample: Dict, coords: List[str], wave_type: str) -> Dict:
    """Create a sample that verifies coordinate accuracy"""
    if len(coords) < 3:
        return None
        
    return {
        "image": original_sample["image"],
        "conversations": [
            {
                "from": "human", 
                "value": f"<image>Verify the accuracy of these {wave_type} wave coordinates with 2ms precision per pixel: x1={coords[0]}, x2={coords[1]}, x3={coords[2]}"
            },
            {
                "from": "gpt",
                "value": f"Verified {wave_type} coordinates: <points x1=\"{coords[0]}\" x2=\"{coords[1]}\" x3=\"{coords[2]}\" alt=\"{wave_type}\">Coordinates confirmed - {wave_type}</points>"
            }
        ]
    }

def create_precision_boundary_sample(original_sample: Dict, coords: List[str], wave_type: str) -> Dict:
    """Create a sample with sub-pixel precision boundaries"""
    if len(coords) < 3:
        return None
    
    # Add small random variations for sub-pixel precision training
    precise_coords = [
        f"{float(coords[0]) + random.uniform(-0.5, 0.5):.1f}",
        f"{float(coords[1]) + random.uniform(-0.5, 0.5):.1f}", 
        f"{float(coords[2]) + random.uniform(-0.5, 0.5):.1f}"
    ]
    
    return {
        "image": original_sample["image"],
        "conversations": [
            {
                "from": "human",
                "value": "<image>Identify wave boundaries with sub-pixel precision. Use decimal coordinates for exact boundaries."
            },
            {
                "from": "gpt",
                "value": f"Sub-pixel precision boundaries: <points x1=\"{precise_coords[0]}\" x2=\"{precise_coords[1]}\" x3=\"{precise_coords[2]}\" alt=\"{wave_type}\">{wave_type}</points>"
            }
        ]
    }

def create_temporal_context_sample(original_sample: Dict) -> Dict:
    """Create a sample that teaches temporal context understanding"""
    return {
        "image": original_sample["image"],
        "conversations": [
            {
                "from": "human",
                "value": "<image>Analyze this 2-second ECG segment. Identify all waves with their exact temporal positions. Each pixel represents 2ms."
            },
            {
                "from": "gpt", 
                "value": "Analyzing 2-second ECG with temporal precision:\n- Time range: 0-2000ms (0-999 pixels)\n- Sampling: 500Hz = 1 sample per pixel\n- Resolution: 2ms per pixel\nIdentifying waves with precise timing..."
            }
        ]
    }

def enhance_ecg_dataset(input_file: str, output_file: str, augmentation_factor: float = 2.0):
    """
    Enhance ECG dataset with coordinate precision training
    
    Args:
        input_file: Path to original dataset JSON
        output_file: Path to save enhanced dataset
        augmentation_factor: Multiplier for dataset size (2.0 = double size)
    """
    print(f"📊 Loading dataset from: {input_file}")
    
    with open(input_file, 'r') as f:
        original_data = json.load(f)
    
    enhanced_data = []
    
    for i, sample in enumerate(original_data):
        if i % 100 == 0:
            print(f"Processing sample {i+1}/{len(original_data)}")
            
        # Always include original sample
        enhanced_data.append(sample)
        
        # Extract coordinates from conversations
        for convo in sample.get('conversations', []):
            if convo.get('from') == 'gpt' and '<points' in convo.get('value', ''):
                coords = extract_coordinates_from_text(convo['value'])
                
                # Determine wave type
                wave_type = "wave"
                if 'alt="' in convo['value']:
                    alt_match = re.search(r'alt="([^"]+)"', convo['value'])
                    if alt_match:
                        wave_type = alt_match.group(1)
                
                if len(coords) >= 3:
                    # Add temporal mapping sample
                    temporal_sample = create_temporal_mapping_sample(sample, coords)
                    if temporal_sample:
                        enhanced_data.append(temporal_sample)
                    
                    # Add coordinate verification sample
                    verification_sample = create_coordinate_verification_sample(sample, coords, wave_type)
                    if verification_sample:
                        enhanced_data.append(verification_sample)
                    
                    # Add precision boundary sample (with probability)
                    if random.random() < 0.7:  # 70% chance
                        precision_sample = create_precision_boundary_sample(sample, coords, wave_type)
                        if precision_sample:
                            enhanced_data.append(precision_sample)
        
        # Add temporal context sample (with probability)
        if random.random() < 0.3:  # 30% chance
            context_sample = create_temporal_context_sample(sample)
            enhanced_data.append(context_sample)
    
    # Trim to desired augmentation factor
    target_size = int(len(original_data) * augmentation_factor)
    if len(enhanced_data) > target_size:
        # Keep all original samples + random selection of augmented samples
        augmented_samples = enhanced_data[len(original_data):]
        random.shuffle(augmented_samples)
        enhanced_data = original_data + augmented_samples[:target_size - len(original_data)]
    
    # Save enhanced dataset
    print(f"💾 Saving enhanced dataset to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    print(f"✅ Enhancement complete!")
    print(f"   Original samples: {len(original_data)}")
    print(f"   Enhanced samples: {len(enhanced_data)}")
    print(f"   Augmentation factor: {len(enhanced_data) / len(original_data):.1f}x")

def main():
    parser = argparse.ArgumentParser(description="Enhance ECG dataset for coordinate precision")
    parser.add_argument("--input", required=True, help="Input dataset JSON file")
    parser.add_argument("--output", required=True, help="Output enhanced dataset JSON file")
    parser.add_argument("--factor", type=float, default=2.0, help="Augmentation factor (default: 2.0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file {args.input} not found")
        return
    
    enhance_ecg_dataset(args.input, args.output, args.factor)

if __name__ == "__main__":
    main() 