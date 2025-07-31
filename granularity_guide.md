# 🎯 ECG Coordinate Granularity Guide

## 📊 **Granularity Options Explained**

Your original concern about 1000-pixel granularity is valid! Here's a comprehensive analysis:

### **Granularity Comparison Table**

| Granularity | Width | Samples/Pixel | Time/Pixel | Coordinate Range | Model Difficulty |
|-------------|-------|---------------|------------|------------------|------------------|
| **Ultra-fine** | 1000px | 1 sample | 2ms | 0-999 | Very Hard 😰 |
| **Fine** | 500px | 2 samples | 4ms | 0-499 | Moderate 😊 |
| **Medium** | 250px | 4 samples | 8ms | 0-249 | Easy 😄 |
| **Coarse** | 200px | 5 samples | 10ms | 0-199 | Very Easy 😎 |

## 🤔 **Why 1000 Pixels Might Be Too Much**

### **Model Perspective:**
- **Attention span**: VLMs have limited attention for fine details
- **Coordinate precision**: Distinguishing pixel 347 vs 348 is extremely difficult
- **Training stability**: Too many coordinate options can cause confusion
- **Overfitting risk**: Model might memorize specific pixel patterns

### **Clinical Perspective:**
- **Medical accuracy**: 2ms precision might exceed clinical needs
- **Measurement error**: Manual annotations have inherent ±2-4ms uncertainty  
- **Device variability**: ECG machines have ~4-8ms sampling precision in practice

## 🎯 **Recommended Granularity Choice**

### **🥇 BEST: Fine Granularity (500px, 4ms/pixel)**

**Why Fine is Optimal:**
- ✅ **Clinically relevant**: 4ms precision is excellent for ECG analysis
- ✅ **Model-friendly**: 500 coordinates is manageable for VLMs
- ✅ **Training efficient**: Faster convergence, more stable
- ✅ **Medical standard**: Matches typical ECG measurement precision
- ✅ **Perfect balance**: Not too coarse, not too fine

**Clinical Context:**
- P wave duration: ~80-100ms → 20-25 pixels
- QRS duration: ~80-120ms → 20-30 pixels  
- T wave duration: ~160-200ms → 40-50 pixels
- **Result**: Each wave spans 20+ pixels → plenty of precision!

### **🥈 Alternative: Medium Granularity (250px, 8ms/pixel)**

**When to Use Medium:**
- Limited training time/resources
- Proof-of-concept development
- Still clinically acceptable (8ms precision)
- Faster training and inference

## 📈 **Expected Accuracy by Granularity**

### **Ultra-fine (1000px, 2ms/pixel):**
- **Target accuracy**: ±1-2 pixels (±2-4ms)
- **Training difficulty**: Very hard
- **Training time**: Longest
- **Clinical benefit**: Minimal over fine

### **Fine (500px, 4ms/pixel):** ⭐ **RECOMMENDED**
- **Target accuracy**: ±1-2 pixels (±4-8ms)  
- **Training difficulty**: Moderate
- **Training time**: Reasonable
- **Clinical benefit**: Excellent

### **Medium (250px, 8ms/pixel):**
- **Target accuracy**: ±1-2 pixels (±8-16ms)
- **Training difficulty**: Easy
- **Training time**: Fast  
- **Clinical benefit**: Good

### **Coarse (200px, 10ms/pixel):**
- **Target accuracy**: ±1-2 pixels (±10-20ms)
- **Training difficulty**: Very easy
- **Training time**: Fastest
- **Clinical benefit**: Acceptable for some applications

## 🔬 **Technical Implementation**

### **Perfect 1:1 Mapping Achieved:**
```python
# Example for Fine granularity:
# 1000 samples → 500 pixels (2 samples per pixel)
# Each pixel represents exactly 4ms of ECG time
# Coordinate 125 = 500ms from start
```

### **Coordinate Scaling:**
```python
# Original sample index → Pixel coordinate
def scale_coordinate(sample_idx, samples_per_pixel):
    return int(sample_idx / samples_per_pixel)

# Examples:
# Ultra-fine: sample 400 → pixel 400 (no scaling)
# Fine: sample 400 → pixel 200 (÷2 scaling)  
# Medium: sample 400 → pixel 100 (÷4 scaling)
```

## 📊 **Training Recommendations**

### **For Fine Granularity (Recommended):**
```bash
# Use the fine granularity setting
CHOSEN_GRANULARITY = "fine"

# Expected results:
# - Image: 500×224 pixels
# - Coordinate range: 0-499
# - Temporal resolution: 4ms/pixel
# - Training time: ~2-3x faster than ultra-fine
# - Accuracy: ±4-8ms (excellent for ECG)
```

### **Training Parameters by Granularity:**
- **Ultra-fine**: Smallest batches, lowest LR, most epochs
- **Fine**: Balanced parameters (recommended)
- **Medium**: Larger batches, higher LR, fewer epochs  
- **Coarse**: Fastest training, highest LR

## 🎯 **Action Plan**

1. **Start with Fine Granularity** (500px, 4ms/pixel)
2. **Run the fixed script** with `CHOSEN_GRANULARITY = "fine"`
3. **Train using adaptive script** (automatically optimizes for fine)
4. **Evaluate results** - should achieve ±4-8ms accuracy
5. **Consider Medium** if you need faster training
6. **Avoid Ultra-fine** unless you specifically need sub-4ms precision

## 💡 **Key Insight**

**More pixels ≠ Better results!**

Fine granularity (500px, 4ms/pixel) provides:
- ✅ Clinically excellent precision
- ✅ Model-friendly coordinate space
- ✅ Faster, more stable training
- ✅ Better generalization
- ✅ Practical ECG analysis precision

The sweet spot is **500 pixels with 4ms temporal resolution** - perfect for medical-grade ECG coordinate detection! 🎯 