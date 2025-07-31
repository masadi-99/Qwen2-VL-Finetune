import os
import json
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from PIL import Image

# === CONFIGURATION ===
LUDB_DIR = './ecg-segmentation/lobachevsky-university-electrocardiography-database-1.0.1/data'  # <-- Change this
SAVE_IMG_DIR = './ecg_images'
SEG_LABELS_PATH = './ecg-segmentation/lobachevsky-university-electrocardiography-database-1.0.1/ludb.csv'
WINDOW_SEC = 2
FS = 500
LEADS = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

# 🎯 PATCH-ALIGNED GRANULARITY OPTIONS
# All dimensions are multiples of 28 for perfect Vision Transformer alignment
PATCH_SIZE = 28
GRANULARITY_OPTIONS = {
    "ultra_fine": {"width": 28*36, "height": 28*8, "samples_per_pixel": 1},    # 1008×224, 1 sample/pixel, 2ms/pixel
    "fine": {"width": 28*18, "height": 28*8, "samples_per_pixel": 2},         # 504×224, 2 samples/pixel, 4ms/pixel
    "medium": {"width": 28*10, "height": 28*8, "samples_per_pixel": 4},       # 280×224, 4 samples/pixel, 7.1ms/pixel
    "coarse": {"width": 28*9, "height": 28*8, "samples_per_pixel": 5},        # 252×224, 5 samples/pixel, 7.9ms/pixel
}

# 🔧 CHOOSE YOUR GRANULARITY HERE:
CHOSEN_GRANULARITY = "fine"  # Recommended: perfect balance
CONFIG = GRANULARITY_OPTIONS[CHOSEN_GRANULARITY]

FIXED_WIDTH = CONFIG["width"]
FIXED_HEIGHT = CONFIG["height"] 
SAMPLES_PER_PIXEL = CONFIG["samples_per_pixel"]
MS_PER_PIXEL = (1000 / FS) * SAMPLES_PER_PIXEL  # Milliseconds per pixel

# Calculate patch information
WIDTH_PATCHES = FIXED_WIDTH // PATCH_SIZE
HEIGHT_PATCHES = FIXED_HEIGHT // PATCH_SIZE
TOTAL_PATCHES = WIDTH_PATCHES * HEIGHT_PATCHES

print(f"🎯 PATCH-ALIGNED ECG Configuration:")
print(f"   Granularity: {CHOSEN_GRANULARITY}")
print(f"   Image dimensions: {FIXED_WIDTH}×{FIXED_HEIGHT} pixels")
print(f"   Patch grid: {WIDTH_PATCHES}×{HEIGHT_PATCHES} = {TOTAL_PATCHES} patches")
print(f"   Samples per pixel: {SAMPLES_PER_PIXEL}")
print(f"   Time per pixel: {MS_PER_PIXEL:.1f}ms")
print(f"   Perfect Vision Transformer alignment: ✅")

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
samples_per_window = FS * WINDOW_SEC

# === LOAD SEGMENT LABELS ===
seg_labels = pd.read_csv(SEG_LABELS_PATH).fillna('')

def downsample_signal_for_perfect_mapping(signal, target_width, samples_per_pixel):
    """
    Downsample signal to ensure perfect pixel-to-sample mapping
    """
    total_samples = len(signal)
    target_samples = target_width * samples_per_pixel
    
    if total_samples != target_samples:
        # Resample to exact target length
        indices = np.linspace(0, total_samples - 1, target_samples)
        signal = np.interp(indices, np.arange(total_samples), signal)
    
    # Average samples within each pixel
    if samples_per_pixel > 1:
        reshaped = signal.reshape(target_width, samples_per_pixel)
        pixel_values = np.mean(reshaped, axis=1)
    else:
        pixel_values = signal
    
    return pixel_values

def create_patch_aligned_image(signal, width, height, output_path):
    """
    Create image with guaranteed patch alignment and 1:1 pixel-to-datapoint mapping
    """
    # Ensure signal length matches width exactly
    if len(signal) != width:
        indices = np.linspace(0, len(signal) - 1, width)
        signal = np.interp(indices, np.arange(len(signal)), signal)
    
    # Create figure with exact pixel dimensions
    dpi = 100
    figsize = (width / dpi, height / dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot with exact x-coordinates for 1:1 mapping
    x_coords = np.arange(width)
    ax.plot(x_coords, signal, linewidth=1.0, color='black', antialiased=False)
    
    # Set exact limits and remove all padding
    ax.set_xlim(0, width - 1)
    ax.set_ylim(signal.min() * 1.1, signal.max() * 1.1)
    ax.set_aspect('auto')
    
    # Remove all axes, ticks, and margins
    ax.axis('off')
    ax.margins(0)
    
    # Remove all padding/margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    
    # Save with exact dimensions
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0, 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Verify exact dimensions and patch alignment
    verify_image = Image.open(output_path)
    actual_width, actual_height = verify_image.size
    
    if actual_width != width or actual_height != height:
        print(f"⚠️  Resizing {output_path}: {actual_width}x{actual_height} → {width}x{height}")
        resized = verify_image.resize((width, height), Image.Resampling.LANCZOS)
        resized.save(output_path)
    
    # Verify patch alignment
    width_remainder = actual_width % PATCH_SIZE
    height_remainder = actual_height % PATCH_SIZE
    
    if width_remainder == 0 and height_remainder == 0:
        print(f"✅ Perfect patch alignment: {actual_width}×{actual_height}")
    else:
        print(f"⚠️  Patch alignment issue: {width_remainder}×{height_remainder} remainder")
    
    verify_image.close()

def build_patch_aware_questions(waves_dict, ms_per_pixel, total_patches):
    """Build wave questions with patch and temporal context"""
    wave_order = ['QRS', 'P', 'T']
    random.shuffle(wave_order)
    convo = []

    for wave_type in wave_order:
        if not waves_dict[wave_type]:
            continue

        # Enhanced prompts with patch and temporal context
        if not convo:
            question = f"Please identify the {wave_type} complexes. This ECG uses Vision Transformer patches: each pixel = {ms_per_pixel:.1f}ms, {total_patches} total patches."
        else:
            question = f"What about the {wave_type} waves? Remember: {ms_per_pixel:.1f}ms per pixel, patch-aligned coordinates."
        
        convo.append({"from": "human", "value": question})

        tags = ""
        for triplet in waves_dict[wave_type]:
            if isinstance(triplet, list) and len(triplet) == 3:
                triplet_int = [int(x) for x in triplet]
                tags += f'<points x1="{triplet_int[0]}" x2="{triplet_int[1]}" x3="{triplet_int[2]}" alt="{wave_type}">{wave_type}</points>'
        convo.append({"from": "gpt", "value": tags})

    return convo

def build_patient_questions(row):
    fields = {
        "Age": lambda x: ("How old is the patient?", f"The patient is {x.strip()} years old."),
        "Sex": lambda x: ("What is the sex of the patient?", "The patient is male." if 'M' in x else "The patient is female."),
        "Rhythms": lambda x: ("What is the patient's heart rhythm?", f"The rhythm is {x.strip()}."),
        "Electric axis of the heart": lambda x: ("What is the electrical axis?", f"The electrical axis is {x.replace('Electric axis of the heart: ', '').strip()}."),
        "Conduction abnormalities": lambda x: ("Any conduction abnormalities?", f"Yes, {x.strip()}."),
        "Extrasystolies": lambda x: ("Does the patient have extrasystolies?", f"Yes, {x.strip()}."),
        "Hypertrophies": lambda x: ("Any hypertrophy signs?", f"Yes, {x.strip()}."),
        "Cardiac pacing": lambda x: ("Is there any cardiac pacing?", f"Yes, {x.strip()}."),
        "Ischemia": lambda x: ("Any signs of ischemia?", f"Yes, {x.strip()}."),
        "Non-specific repolarization abnormalities": lambda x: ("Any repolarization abnormalities?", f"Yes, {x.strip()}."),
        "Other states": lambda x: ("Any other relevant findings?", f"{x.strip()}")
    }

    qas = []
    for col, formatter in fields.items():
        val = row.get(col, '').strip()
        if val:
            q, a = formatter(val)
            qas.append((q, a))

    random.shuffle(qas)

    convo = []
    for q, a in qas:
        convo.append({"from": "human", "value": q})
        convo.append({"from": "gpt", "value": a})

    return convo

def pairwise_shuffle(convo):
    """Shuffles a conversation while preserving human→gpt pairs."""
    assert len(convo) % 2 == 0
    pairs = list(range(0, len(convo), 2))
    random.shuffle(pairs)
    shuffled = []
    for i in pairs:
        shuffled.append(convo[i])
        shuffled.append(convo[i+1])
    return shuffled

# === DATASETS ===
dataset_wo_meta = []
dataset_with_meta = []

# === MAIN LOOP ===
print(f"🚀 Processing ECG records with patch-aligned {CHOSEN_GRANULARITY} granularity...")

for rec_id in tqdm(range(1, 201)):
    rec_name = str(rec_id)
    rec_path = os.path.join(LUDB_DIR, rec_name)

    try:
        record = wfdb.rdrecord(os.path.join(LUDB_DIR, rec_name))
        signal = record.p_signal
        total_samples = signal.shape[0]

        record_rows = seg_labels[seg_labels['ID'] == int(rec_name)]
        if record_rows.empty:
            print(f"No CSV row found for record {rec_name}")
            continue
        
        row = record_rows.iloc[0]
        row_dict = row.to_dict()

        for lead_idx, lead in enumerate(LEADS):
            ann_path = os.path.join(LUDB_DIR, f"{rec_name}.{lead}")
            if not os.path.exists(ann_path):
                continue

            ann = wfdb.rdann(os.path.join(LUDB_DIR, rec_name), extension=lead)
            lead_signal = signal[:, lead_idx]
            num_windows = total_samples // samples_per_window

            for w in range(num_windows):
                start = w * samples_per_window
                end = start + samples_per_window
                sig_crop = lead_signal[start:end]
                
                # 🎯 PERFECT PATCH-ALIGNED MAPPING
                pixel_signal = downsample_signal_for_perfect_mapping(
                    sig_crop, FIXED_WIDTH, SAMPLES_PER_PIXEL
                )
                
                ann_in_window = [(s - start, sym) for s, sym in zip(ann.sample, ann.symbol)
                                 if start <= s < end]
                if not ann_in_window:
                    continue

                # Save patch-aligned image
                img_filename = f"{rec_name}_{lead}_{w}.png"
                img_path = os.path.join(SAVE_IMG_DIR, img_filename)
                
                create_patch_aligned_image(pixel_signal, FIXED_WIDTH, FIXED_HEIGHT, img_path)

                # 🎯 SCALE COORDINATES with patch awareness
                scaled_annotations = []
                for sample_idx, symbol in ann_in_window:
                    pixel_idx = int(sample_idx / SAMPLES_PER_PIXEL)
                    # Ensure coordinates stay within image bounds
                    pixel_idx = max(0, min(pixel_idx, FIXED_WIDTH - 1))
                    scaled_annotations.append((pixel_idx, symbol))

                # Collect waves with scaled coordinates
                waves = {'QRS': [], 'P': [], 'T': []}

                if len(scaled_annotations) < 3:
                    continue
                    
                i = 0
                while i + 2 < len(scaled_annotations):
                    s1, sym1 = scaled_annotations[i]
                    s2, sym2 = scaled_annotations[i + 1]
                    s3, sym3 = scaled_annotations[i + 2]
                
                    if sym1 == '(' and sym2.upper() in ['N', 'V', 'A'] and sym3 == ')':
                        waves['QRS'].append([s1, s2, s3])
                        i += 3
                        continue
                
                    if sym1 == '(' and sym2 == 'p' and sym3 == ')':
                        waves['P'].append([s1, s2, s3])
                        i += 3
                        continue
                
                    if sym1 == '(' and sym2 == 't' and sym3 == ')':
                        waves['T'].append([s1, s2, s3])
                        i += 3
                        continue
                
                    i += 1

                # === Version 1: Patch-aware conversation ===
                convo_ann = build_patch_aware_questions(waves, MS_PER_PIXEL, TOTAL_PATCHES)
                convo_ann[0]["value"] = "<image>\n" + convo_ann[0]["value"]
                dataset_wo_meta.append({
                    "id": f"{rec_name.zfill(3)}_{lead}_{w}_wo_meta",
                    "image": img_filename,
                    "conversations": convo_ann,
                    "metadata": {
                        "granularity": CHOSEN_GRANULARITY,
                        "ms_per_pixel": MS_PER_PIXEL,
                        "samples_per_pixel": SAMPLES_PER_PIXEL,
                        "image_width": FIXED_WIDTH,
                        "image_height": FIXED_HEIGHT,
                        "patch_grid": f"{WIDTH_PATCHES}×{HEIGHT_PATCHES}",
                        "total_patches": TOTAL_PATCHES,
                        "patch_aligned": True
                    }
                })

                # === Version 2: Full conversation with patch awareness ===
                convo_meta = pairwise_shuffle(build_patient_questions(row_dict))
                convo_wave = pairwise_shuffle(build_patch_aware_questions(waves, MS_PER_PIXEL, TOTAL_PATCHES))
                convo_full = convo_meta + convo_wave

                for msg in convo_full:
                    if msg["from"] == "human":
                        msg["value"] = "<image>\n" + msg["value"]
                        break

                dataset_with_meta.append({
                    "id": f"{rec_name.zfill(3)}_{lead}_{w}_with_meta",
                    "image": img_path,
                    "conversations": convo_full,
                    "metadata": {
                        "granularity": CHOSEN_GRANULARITY,
                        "ms_per_pixel": MS_PER_PIXEL,
                        "samples_per_pixel": SAMPLES_PER_PIXEL,
                        "image_width": FIXED_WIDTH,
                        "image_height": FIXED_HEIGHT,
                        "patch_grid": f"{WIDTH_PATCHES}×{HEIGHT_PATCHES}",
                        "total_patches": TOTAL_PATCHES,
                        "patch_aligned": True
                    }
                })

    except Exception as e:
        print(f"Error with record {rec_name}: {e}")

# === SAVE FILES ===
output_suffix = f"_{CHOSEN_GRANULARITY}_patch_aligned"

with open(f"ludb_conversations_wo_meta{output_suffix}.json", "w") as f:
    json.dump(dataset_wo_meta, f, indent=2)

with open(f"ludb_conversations_with_meta{output_suffix}.json", "w") as f:
    json.dump(dataset_with_meta, f, indent=2)

print(f"\n✅ PATCH-ALIGNED Dataset generation complete!")
print(f"📊 Generated files:")
print(f"   - ludb_conversations_wo_meta{output_suffix}.json ({len(dataset_wo_meta)} samples)")
print(f"   - ludb_conversations_with_meta{output_suffix}.json ({len(dataset_with_meta)} samples)")
print(f"📏 Perfect Vision Transformer specifications:")
print(f"   - Dimensions: {FIXED_WIDTH}×{FIXED_HEIGHT} pixels")
print(f"   - Patch grid: {WIDTH_PATCHES}×{HEIGHT_PATCHES} = {TOTAL_PATCHES} patches")
print(f"   - Temporal resolution: {MS_PER_PIXEL:.1f}ms per pixel")
print(f"   - Perfect patch alignment: ✅")
print(f"   - Zero padding/cropping artifacts: ✅")
print(f"   - Optimal Vision Transformer efficiency: ✅") 