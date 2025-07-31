"""
Text-Only GRPO Trainer
Based on QwenGRPOTrainer but adapted for text-only models without vision components
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torch.utils.data import Dataset, IterableDataset
from trl import GRPOConfig
from trl.trainer.callbacks import TrainerCallback
from trl.trainer.grpo_trainer import GRPOTrainer


class QwenGRPOTextTrainer(GRPOTrainer):
    """
    Text-only GRPO Trainer for Qwen2.5 models
    Removes all vision-related processing and focuses on text generation
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[Callable, list[Callable]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None
    ):
        
        # Store reward functions first
        if isinstance(reward_funcs, list):
            self.reward_funcs = reward_funcs
        else:
            self.reward_funcs = [reward_funcs]
        
        # Initialize the parent GRPO trainer with reward_funcs
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        
        # Text-only models use tokenizer instead of processor
        self.tokenizer = processing_class
    

    
 