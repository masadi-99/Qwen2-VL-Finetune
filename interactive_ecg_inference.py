#!/usr/bin/env python3
"""
Interactive ECG Inference Script
Simple interface for quick testing with patch-aligned models
"""

import os
import torch
import json
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path
from qwen_vl_utils import process_vision_info

def process_vision_info_aligned(messages):
    """Process vision info while preserving dimensions"""
    modified_conversation = []
    
    for msg in messages:
        modified_msg = msg.copy()
        if 'content' in modified_msg:
            modified_content = []
            for item in modified_msg['content']:
                if item.get('type') == 'image':
                    img = Image.open(item['image'])
                    orig_width, orig_height = img.size
                    
                    new_item = item.copy()
                    new_item['resized_width'] = orig_width
                    new_item['resized_height'] = orig_height
                    new_item.pop('min_pixels', None)
                    new_item.pop('max_pixels', None)
                    
                    print(f"🎯 Image: {orig_width}×{orig_height} pixels")
                    
                    # Check patch alignment
                    if orig_width % 28 == 0 and orig_height % 28 == 0:
                        patches = (orig_width // 28) * (orig_height // 28)
                        print(f"✅ Perfect patch alignment: {patches} patches")
                    else:
                        print(f"⚠️  Non-aligned dimensions")
                    
                    modified_content.append(new_item)
                else:
                    modified_content.append(item)
            modified_msg['content'] = modified_content
        modified_conversation.append(modified_msg)
    
    return process_vision_info(modified_conversation)

def main():
    print("🎯 INTERACTIVE ECG INFERENCE")
    print("=" * 50)
    
    # Model setup
    model_path = input("Enter model path: ").strip()
    if not model_path:
        model_path = "./output/ludb_precision_model"  # Default
    
    print(f"Loading model from: {model_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = get_model_name_from_path(model_path)
    
    try:
        processor, model = load_pretrained_model(
            model_base="Qwen/Qwen2.5-VL-3B-Instruct",
            model_path=model_path,
            device_map=device,
            model_name=model_name,
            use_flash_attn=True
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\n🎯 Interactive Mode")
    print("Commands:")
    print("  - Enter image path and question")
    print("  - Type 'quit' to exit")
    print("  - Type 'help' for guidance")
    
    while True:
        print("\n" + "-" * 50)
        
        # Get user input
        command = input(">>> ").strip()
        
        if command.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if command.lower() == 'help':
            print("""
🎯 Usage Examples:
  Image path: ./ecg_images/001_i_0.png
  Question: Please identify the P complexes.
  
🎯 Tips:
  - Use patch-aligned images (504×224 recommended)
  - Ask for specific wave types: P, QRS, T
  - Images should be in proper ECG format
  
🎯 Expected Output:
  <points x1="123" x2="145" x3="167" alt="P">P</points>
            """)
            continue
        
        # Get image path
        if os.path.exists(command):
            image_path = command
        else:
            image_path = input("Image path: ").strip()
            
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
        
        # Get question
        question = input("Question: ").strip()
        if not question:
            question = "Please identify the P, QRS, and T complexes."
        
        # Add temporal context
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # Estimate granularity based on width
            if width == 504:
                ms_per_pixel = 4.0
                granularity = "fine"
            elif width == 1008:
                ms_per_pixel = 2.0
                granularity = "ultra_fine"
            elif width == 280:
                ms_per_pixel = 7.1
                granularity = "medium"
            else:
                ms_per_pixel = 2000 / width  # Estimate
                granularity = "custom"
            
            enhanced_question = f"""
🎯 ECG Analysis Context:
- Image granularity: {granularity} ({ms_per_pixel:.1f}ms per pixel)
- Coordinate precision: Each pixel = {ms_per_pixel:.1f} milliseconds

{question}

📏 Provide precise pixel coordinates (x1, x2, x3) for each detected wave.
"""
        except Exception:
            enhanced_question = question
        
        # Prepare conversation
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": enhanced_question}
            ]
        }]
        
        print("\n🔄 Processing...")
        
        try:
            # Process with patch alignment
            prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info_aligned(conversation)
            
            inputs = processor(
                text=[prompt], 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True, 
                return_tensors="pt", 
                do_resize=False  # Critical for coordinate preservation
            ).to(device)
            
            # Generate response
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True
                )
            
            response = processor.batch_decode(
                output_ids[:, inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            )[0].strip()
            
            print(f"\n🎯 Model Response:")
            print(f"<<< {response}")
            
            # Extract coordinates for summary
            import re
            coords = re.findall(r'x1="(\d+)"\s+x2="(\d+)"\s+x3="(\d+)"\s+alt="([^"]+)"', response)
            if coords:
                print(f"\n📊 Detected Coordinates:")
                for x1, x2, x3, wave_type in coords:
                    time_start = int(x1) * ms_per_pixel
                    time_center = int(x2) * ms_per_pixel
                    time_end = int(x3) * ms_per_pixel
                    print(f"   {wave_type}: pixels {x1}-{x2}-{x3} → {time_start:.1f}-{time_center:.1f}-{time_end:.1f}ms")
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    main() 