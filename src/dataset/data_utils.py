import re
import torch

from qwen_vl_utils import process_vision_info

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
)


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel, width, height):
    # COMPREHENSIVE FIX: Completely bypass resizing to preserve coordinates
    from PIL import Image
    
    # Load and preserve original image
    img = Image.open(image_path)
    orig_width, orig_height = img.size
    
    print(f"DEBUG: Processing {image_path} - Original: {orig_width}x{orig_height}")
    
    content = {
        "type": "image", 
        "image": image_path,
        # COMPLETELY REMOVED min_pixels and max_pixels
        # Force explicit dimensions to prevent ANY automatic resizing
        "resized_width": orig_width,
        "resized_height": orig_height
    }

    # Always use original dimensions, ignore any width/height parameters
    if width is not None and height is not None:
        print(f"WARNING: Ignoring resize request {width}x{height}, using original {orig_width}x{orig_height}")
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, width, height, fps, nframes):
    # FIXED: Preserve original video dimensions to prevent coordinate misalignment
    
    content = {
        "type": "video", 
        "video": video_path,
        # REMOVED min_pixels and max_pixels to prevent automatic resizing
        # "min_pixels": min_pixels,
        # "max_pixels": max_pixels,
    }

    if nframes is not None:
        content["nframes"] = nframes
    else:
        content["fps"] = fps

    # Force original dimensions for videos too (if resizing requested)
    if width is not None and height is not None:
        print(f"WARNING: Video resize request ignored for coordinate preservation")
        # Don't set resized dimensions to prevent scaling
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs

def samples_per_class_from_ids(label_ids, num_classes):
    
    counts = torch.bincount(
        torch.as_tensor(label_ids, dtype=torch.long),
        minlength=num_classes
    )
    
    return counts.tolist()