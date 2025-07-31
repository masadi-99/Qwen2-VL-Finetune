# Core text-only GRPO imports (minimal dependencies)
from .grpo_text_dataset import GRPOTextDataset, make_grpo_text_data_module

# Optional imports with graceful fallback for full functionality
try:
    from .sft_dataset import SupervisedDataset, make_supervised_data_module
    from .dpo_dataset import DPODataset, make_dpo_data_module
    from .grpo_dataset import GRPODataset, make_grpo_data_module
    from .cls_dataset import ClassificationDataset, make_classification_data_module
    
    __all__ = [
        # Text-only GRPO (always available)
        "GRPOTextDataset",
        "make_grpo_text_data_module",
        # Full functionality (optional)
        "SupervisedDataset", 
        "make_supervised_data_module",
        "DPODataset", 
        "make_dpo_data_module",
        "GRPODataset", 
        "make_grpo_data_module",
        "ClassificationDataset", 
        "make_classification_data_module",
    ]
except ImportError as e:
    # Minimal text-only GRPO functionality
    __all__ = [
        "GRPOTextDataset",
        "make_grpo_text_data_module",
    ]