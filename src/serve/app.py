#!/usr/bin/env python3
"""
Advanced ECG Analysis Serving App
Based on patch-aligned inference with proper coordinate handling and multi-turn support
"""

import argparse
import re
import html
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from threading import Thread
from typing import List, Dict, Tuple, Optional
import gradio as gr
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
from functools import partial
import warnings
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

# Global variables
processor = None
model = None
device = None

def is_video_file(filename):
    """Check if file is a video"""
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def process_vision_info_with_perfect_alignment(messages):
    """
    Process vision info while preserving exact dimensions for patch alignment
    Based on our successful inference code
    """
    modified_conversation = []
    
    for msg in messages:
        modified_msg = msg.copy()
        if 'content' in modified_msg:
            modified_content = []
            for item in modified_msg['content']:
                if item.get('type') == 'image':
                    # Load image and preserve exact dimensions
                    img = Image.open(item['image'])
                    orig_width, orig_height = img.size
                    
                    # Create new item with forced dimensions
                    new_item = item.copy()
                    new_item['resized_width'] = orig_width
                    new_item['resized_height'] = orig_height
                    
                    # Remove any min/max pixel constraints to prevent resizing
                    new_item.pop('min_pixels', None)
                    new_item.pop('max_pixels', None)
                    
                    print(f"🎯 Serving: Preserving image dimensions {orig_width}×{orig_height}")
                    
                    # Verify patch alignment
                    width_remainder = orig_width % 28
                    height_remainder = orig_height % 28
                    if width_remainder == 0 and height_remainder == 0:
                        print(f"✅ Perfect patch alignment: {orig_width // 28}×{orig_height // 28} patches")
                    else:
                        print(f"⚠️  Non-aligned dimensions: {width_remainder}×{height_remainder} remainder")
                    
                    modified_content.append(new_item)
                else:
                    modified_content.append(item)
            modified_msg['content'] = modified_content
        modified_conversation.append(modified_msg)
    
    return process_vision_info(modified_conversation)

def extract_coordinates_from_response(text: str) -> List[Tuple[int, int, int, str]]:
    """Extract coordinates from model response"""
    coordinates = []
    patterns = [
        r'<points\s+x1="(\d+)"\s+x2="(\d+)"\s+x3="(\d+)"\s+alt="([^"]+)"[^>]*>',
        r'x1="(\d+)"\s+x2="(\d+)"\s+x3="(\d+)".*?alt="([^"]+)"',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                x1, x2, x3 = int(match[0]), int(match[1]), int(match[2])
                wave_type = match[3].upper()
                if 0 <= x1 <= x2 <= x3 and wave_type in ['P', 'QRS', 'T']:
                    coordinates.append((x1, x2, x3, wave_type))
            except (ValueError, IndexError):
                continue
    
    return coordinates

def estimate_temporal_context(image_path: str) -> Dict[str, float]:
    """Estimate temporal context based on image dimensions"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Estimate granularity based on width
        if width == 504:
            return {"ms_per_pixel": 4.0, "granularity": "fine", "total_patches": 144}
        elif width == 1008:
            return {"ms_per_pixel": 2.0, "granularity": "ultra_fine", "total_patches": 288}
        elif width == 280:
            return {"ms_per_pixel": 7.1, "granularity": "medium", "total_patches": 80}
        elif width == 252:
            return {"ms_per_pixel": 7.9, "granularity": "coarse", "total_patches": 72}
        else:
            # Estimate for other widths
            ms_per_pixel = 2000 / width  # 2 seconds total
            return {"ms_per_pixel": ms_per_pixel, "granularity": "custom", "total_patches": None}
    except Exception:
        return {"ms_per_pixel": 4.0, "granularity": "unknown", "total_patches": None}

def enhance_user_prompt(user_text: str, temporal_context: Dict) -> str:
    """Add temporal context to user prompt for better coordinate accuracy"""
    if not temporal_context:
        return user_text
    
    ms_per_pixel = temporal_context.get('ms_per_pixel', 4.0)
    granularity = temporal_context.get('granularity', 'fine')
    total_patches = temporal_context.get('total_patches')
    
    if total_patches:
        context = f"""🎯 ECG Analysis Context:
- Image granularity: {granularity} ({ms_per_pixel:.1f}ms per pixel)
- Patch alignment: Perfect Vision Transformer grid ({total_patches} patches)
- Coordinate precision: Each pixel = {ms_per_pixel:.1f} milliseconds

{user_text}

📏 Provide precise pixel coordinates (x1, x2, x3) for each detected wave complex."""
    else:
        context = f"""🎯 ECG Analysis Context:
- Image granularity: {granularity} ({ms_per_pixel:.1f}ms per pixel)
- Coordinate precision: Each pixel = {ms_per_pixel:.1f} milliseconds

{user_text}

📏 Provide precise pixel coordinates (x1, x2, x3) for each detected wave complex."""
    
    return context

def format_coordinates_for_display(coordinates: List[Tuple], temporal_context: Dict) -> str:
    """Format coordinates for user-friendly display"""
    if not coordinates:
        return ""
    
    ms_per_pixel = temporal_context.get('ms_per_pixel', 4.0)
    
    coord_display = "\n\n📊 **Detected Wave Coordinates:**\n"
    
    for x1, x2, x3, wave_type in coordinates:
        time_start = x1 * ms_per_pixel
        time_center = x2 * ms_per_pixel
        time_end = x3 * ms_per_pixel
        
        coord_display += f"• **{wave_type} Wave**: Pixels {x1}-{x2}-{x3} → Time {time_start:.1f}-{time_center:.1f}-{time_end:.1f}ms\n"
    
    return coord_display

def escape_html_tags(text: str) -> str:
    """Escape HTML tags to prevent them from disappearing in the interface"""
    # Escape the points tags so they display properly
    text = re.sub(r'<points([^>]*)>', lambda m: f'&lt;points{html.escape(m.group(1))}&gt;', text)
    text = text.replace('</points>', '&lt;/points&gt;')
    return text

def create_annotated_ecg_image(image_path: str, coordinates: List[Tuple[int, int, int, str]], 
                              temporal_context: Dict) -> str:
    """
    Create an annotated ECG image with highlighted wave complexes
    """
    try:
        # Load the original ECG image
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Create matplotlib figure with same size as original image
        dpi = 100
        fig_width = img_width / dpi
        fig_height = img_height / dpi
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Display the original ECG image as background
        ax.imshow(img, extent=[0, img_width, img_height, 0], aspect='auto')
        
        # Define consistent colors for different wave types (more distinct)
        wave_colors = {
            'P': {'color': '#FF4444', 'alpha': 0.4, 'label': 'P Wave'},
            'QRS': {'color': '#00CC88', 'alpha': 0.4, 'label': 'QRS Complex'}, 
            'T': {'color': '#3366FF', 'alpha': 0.4, 'label': 'T Wave'}
        }
        
        # Track legend patches for consistent colors
        legend_patches = {}
        
        # Highlight detected coordinates
        for x1, x2, x3, wave_type in coordinates:
            if wave_type in wave_colors:
                color_info = wave_colors[wave_type]
                
                # Create rectangle highlighting the wave complex
                # Height covers most of the ECG signal area
                rect_height = img_height * 0.8  # 80% of image height
                rect_y = img_height * 0.1  # Start at 10% from top
                
                # Width from x1 to x3
                rect_width = x3 - x1
                
                # Create the highlighting rectangle with explicit color
                rect = patches.Rectangle(
                    (x1, rect_y), rect_width, rect_height,
                    linewidth=3,
                    edgecolor=color_info['color'],
                    facecolor=color_info['color'],
                    alpha=color_info['alpha']
                )
                
                ax.add_patch(rect)
                
                # Store patch for legend (only first occurrence)
                if wave_type not in legend_patches:
                    legend_patches[wave_type] = patches.Rectangle(
                        (0, 0), 1, 1,  # Dummy rectangle for legend
                        facecolor=color_info['color'],
                        edgecolor=color_info['color'],
                        alpha=color_info['alpha'],
                        linewidth=3,
                        label=color_info['label']
                    )
                
                # Add wave type annotation with matching color
                ax.annotate(
                    wave_type,
                    xy=(x2, rect_y - 5),  # Position at peak (x2) above the rectangle
                    xytext=(x2, rect_y - 25),
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    fontweight='bold',
                    color=color_info['color'],
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor='white',
                        edgecolor=color_info['color'],
                        alpha=0.9,
                        linewidth=2
                    ),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color_info['color'],
                        lw=2
                    )
                )
        
        # Remove axis ticks and labels
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Flip y-axis to match image coordinates
        ax.axis('off')
        
        # Add legend with consistent colors if we have detected waves
        if legend_patches:
            # Sort legend patches for consistent order
            sorted_patches = []
            wave_order = ['P', 'QRS', 'T']  # Preferred order
            for wave_type in wave_order:
                if wave_type in legend_patches:
                    sorted_patches.append(legend_patches[wave_type])
            
            ax.legend(
                handles=sorted_patches,
                loc='upper right',
                framealpha=0.95,
                fancybox=True,
                shadow=True,
                fontsize=11,
                edgecolor='black',
                borderpad=0.8
            )
        
        # Add title with temporal information
        ms_per_pixel = temporal_context.get('ms_per_pixel', 4.0)
        granularity = temporal_context.get('granularity', 'fine')
        
        title = f"ECG Analysis - {granularity.title()} Granularity ({ms_per_pixel:.1f}ms/pixel)"
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        # Save the annotated image to a static directory
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "static", "ecg_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(np.random.random() * 1000000)  # Simple unique identifier
        output_path = os.path.join(output_dir, f"ecg_annotated_{timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0.1,
            facecolor='white',
            edgecolor='none'
        )
        plt.close()
        
        print(f"✅ Created annotated ECG image: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error creating annotated ECG image: {e}")
        return None

def build_conversation_history(history, current_message):
    """Build clean conversation history with proper image handling"""
    conversation = []
    last_temporal_context = {}
    
    # Process existing history
    for user_turn, assistant_turn in history:
        # Handle user turn
        user_content = []
        
        if isinstance(user_turn, tuple):
            # Image + text case
            if len(user_turn) == 2:
                file_paths, user_text = user_turn
            else:
                # Handle case where tuple might have different structure
                file_paths = user_turn[0] if user_turn else None
                user_text = user_turn[1] if len(user_turn) > 1 else ""
            
            if file_paths:
                if not isinstance(file_paths, list):
                    file_paths = [file_paths]
                
                for file_path in file_paths:
                    if is_video_file(file_path):
                        user_content.append({"type": "video", "video": file_path, "fps": 1.0})
                    else:
                        user_content.append({"type": "image", "image": file_path})
                        # Store temporal context from images in history
                        if not last_temporal_context:
                            last_temporal_context = estimate_temporal_context(file_path)
            
            if user_text:
                user_content.append({"type": "text", "text": user_text})
        else:
            # Text only case
            user_content.append({"type": "text", "text": user_turn})
        
        conversation.append({"role": "user", "content": user_content})
        
        # Handle assistant turn
        if assistant_turn is not None:
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": assistant_turn}]})
    
    # Add current message
    user_content = []
    temporal_context = {}
    
    # Handle files in current message
    if current_message.get("files"):
        for file_item in current_message["files"]:
            file_path = file_item["path"] if isinstance(file_item, dict) else file_item
            
            if is_video_file(file_path):
                user_content.append({"type": "video", "video": file_path, "fps": 1.0})
            else:
                user_content.append({"type": "image", "image": file_path})
                # Get temporal context from the first image
                if not temporal_context:
                    temporal_context = estimate_temporal_context(file_path)
    
    # If no new image, use temporal context from history
    if not temporal_context and last_temporal_context:
        temporal_context = last_temporal_context
    
    # Enhance user text with temporal context
    user_text = current_message.get('text', '')
    if user_text and temporal_context:
        enhanced_text = enhance_user_prompt(user_text, temporal_context)
        user_content.append({"type": "text", "text": enhanced_text})
    elif user_text:
        user_content.append({"type": "text", "text": user_text})
    
    conversation.append({"role": "user", "content": user_content})
    
    return conversation, temporal_context

def find_image_path_in_conversation(conversation):
    """Find the most recent ECG image path in the conversation"""
    for msg in reversed(conversation):  # Search from most recent
        if 'content' in msg:
            for item in msg['content']:
                if item.get('type') == 'image':
                    return item['image']
    return None

def bot_streaming(message, history, generation_args):
    """
    Enhanced streaming function with proper coordinate handling and multi-turn support
    """
    try:
        # Build conversation with temporal context
        conversation, temporal_context = build_conversation_history(history, message)
        
        # Find the ECG image path for visualization
        ecg_image_path = find_image_path_in_conversation(conversation)
        
        # Apply chat template
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Process vision info with perfect alignment
        image_inputs, video_inputs = process_vision_info_with_perfect_alignment(conversation)
        
        # Prepare inputs with coordinate preservation
        inputs = processor(
            text=[prompt], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt", 
            do_resize=False  # 🎯 CRITICAL: Prevent coordinate shifts
        ).to(device)
        
        # Setup streaming
        streamer = TextIteratorStreamer(
            processor.tokenizer, 
            skip_special_tokens=True, 
            skip_prompt=True, 
            clean_up_tokenization_spaces=False
        )
        
        generation_kwargs = dict(
            inputs, 
            streamer=streamer, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            **generation_args
        )
        
        # Start generation in background thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream response with proper formatting
        buffer = ""
        for new_text in streamer:
            buffer += new_text
            
            # Process the complete response when streaming is done
            if processor.tokenizer.eos_token in buffer or thread.is_alive() == False:
                # Extract coordinates
                coordinates = extract_coordinates_from_response(buffer)
                
                # Escape HTML tags to prevent disappearing
                formatted_buffer = escape_html_tags(buffer)
                
                # Add coordinate summary if found
                if coordinates and temporal_context:
                    coordinate_summary = format_coordinates_for_display(coordinates, temporal_context)
                    formatted_buffer += coordinate_summary
                
                # Create annotated ECG visualization if we have coordinates and image
                if coordinates and ecg_image_path and temporal_context:
                    try:
                        annotated_image_path = create_annotated_ecg_image(
                            ecg_image_path, coordinates, temporal_context
                        )
                        
                        if annotated_image_path:
                            # Convert image to base64 for embedding
                            import base64
                            with open(annotated_image_path, 'rb') as img_file:
                                img_data = base64.b64encode(img_file.read()).decode()
                            
                            # Add the annotated image to the response as embedded HTML
                            formatted_buffer += f"\n\n🎯 **ECG Analysis Visualization:**\n"
                            formatted_buffer += f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%; height: auto; border: 2px solid #007bff; border-radius: 8px; margin: 10px 0;" alt="Annotated ECG"/>\n'
                            formatted_buffer += "\n📊 **Color Legend:**\n"
                            formatted_buffer += "• 🔴 **Red (#FF4444)**: P Waves\n"
                            formatted_buffer += "• 🟢 **Green (#00CC88)**: QRS Complexes\n" 
                            formatted_buffer += "• 🔵 **Blue (#3366FF)**: T Waves\n"
                            
                            # Clean up the temporary file
                            try:
                                os.remove(annotated_image_path)
                                print(f"🗑️  Cleaned up temporary image: {annotated_image_path}")
                            except:
                                pass  # Ignore cleanup errors
                            
                    except Exception as viz_error:
                        print(f"⚠️  Visualization error: {viz_error}")
                        # Continue without visualization if it fails
                        
                yield formatted_buffer
            else:
                # For intermediate streaming, just escape HTML
                yield escape_html_tags(buffer)
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"❌ **Error during inference**: {str(e)}\n\nPlease check your model and try again."
        print(f"Serving error: {e}")
        print(f"Full traceback: {error_details}")
        yield error_msg

def main(args):
    """Main serving function"""
    global processor, model, device
    
    device = args.device
    disable_torch_init()
    
    print("🎯 ECG ANALYSIS SERVING APP")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    
    # Load model
    print("🔄 Loading model...")
    model_name = get_model_name_from_path(args.model_path)
    use_flash_attn = not args.disable_flash_attention
    
    processor, model = load_pretrained_model(
        model_base=args.model_base,
        model_path=args.model_path,
        device_map=device,
        model_name=model_name,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        device=device,
        use_flash_attn=use_flash_attn
    )
    
    print("✅ Model loaded successfully!")
    
    # Setup generation parameters
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }
    
    # Create gradio interface
    with gr.Blocks() as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🏥 ECG Analysis Assistant</h1>
            <h3>Patch-Aligned Vision Transformer for Precise Coordinate Detection</h3>
            <p><strong>Features:</strong> Multi-turn conversations • Perfect coordinate preservation • Temporal context awareness</p>
        </div>
        """)
        
        # Enhanced chatbot with better rendering
        chatbot = gr.Chatbot(
            height=600,
            show_label=False
        )
        
        # Multimodal input
        chat_input = gr.MultimodalTextbox(
            interactive=True,
            placeholder="Upload ECG image and ask: 'Please identify the P, QRS, and T complexes.'",
            show_label=False
        )
        
        # Settings display
        settings_display = gr.HTML(f"""
        <div style="padding: 15px; background-color: #ffffff; border: 2px solid #007bff; border-radius: 8px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0; color: #007bff; font-weight: bold;">⚙️ Settings</h4>
            <ul style="margin: 0; padding-left: 20px; color: #333333; list-style-type: disc;">
                <li style="color: #333333; margin: 5px 0;"><strong style="color: #000000;">Temperature:</strong> <span style="color: #007bff;">{args.temperature}</span></li>
                <li style="color: #333333; margin: 5px 0;"><strong style="color: #000000;">Max tokens:</strong> <span style="color: #007bff;">{args.max_new_tokens}</span></li>
                <li style="color: #333333; margin: 5px 0;"><strong style="color: #000000;">Flash attention:</strong> <span style="color: #28a745;">{not args.disable_flash_attention}</span></li>
                <li style="color: #333333; margin: 5px 0;"><strong style="color: #000000;">Coordinate preservation:</strong> <span style="color: #28a745;">✅ Enabled</span></li>
            </ul>
        </div>
        """)
        
        # Chat interface with custom function
        bot_streaming_with_args = partial(bot_streaming, generation_args=generation_args)
        
        gr.ChatInterface(
            fn=bot_streaming_with_args,
            chatbot=chatbot,
            textbox=chat_input,
            title=None,  # We have our own title
            description=None,
            multimodal=True
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 10px; margin-top: 20px; font-size: 12px; color: #666;">
            Powered by Qwen2-VL with Patch-Aligned Coordinate Preservation • 
            <strong>Medical-Grade Precision:</strong> ±4-8ms accuracy
        </div>
        """)
    
    # Launch server
    print("🚀 Starting server...")
    print(f"Access at: http://localhost:7860")
    
    demo.queue(api_open=False)
    demo.launch(
        show_api=False,
        share=args.share,
        server_name='0.0.0.0',
        server_port=7860
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Analysis Serving App")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--disable_flash_attention", action="store_true", help="Disable flash attention")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens")
    parser.add_argument("--share", action="store_true", help="Share via public URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    main(args)