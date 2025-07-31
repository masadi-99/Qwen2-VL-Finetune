"""
Text-Only GRPO Trainer
Based on QwenGRPOTrainer but adapted for text-only models without vision components
"""

import copy
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torch.utils.data import Dataset, IterableDataset
from trl import GRPOConfig
from trl.trainer.callbacks import TrainerCallback
from trl.trainer.grpo_trainer import GRPOTrainer

from src.train.reward_funcs import RewardFunc


class QwenGRPOTextTrainer(GRPOTrainer):
    """
    Text-only GRPO Trainer for Qwen2.5 models
    Removes all vision-related processing and focuses on text generation
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None
    ):
        
        # Initialize the parent GRPO trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        
        # Store reward functions
        if isinstance(reward_funcs, list):
            self.reward_funcs = reward_funcs
        else:
            self.reward_funcs = [reward_funcs]
        
        # Text-only models use tokenizer instead of processor
        self.tokenizer = processing_class
        
        # Set up reward processing classes
        if reward_processing_classes is None:
            self.reward_processing_classes = [processing_class] * len(self.reward_funcs)
        elif isinstance(reward_processing_classes, list):
            self.reward_processing_classes = reward_processing_classes
        else:
            self.reward_processing_classes = [reward_processing_classes]
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep=None, batch_size=None):
        """
        Get per-token log probabilities for text-only models
        Simplified version without multimodal inputs
        """
        device = input_ids.device
        
        if batch_size is None:
            batch_size = input_ids.shape[0]
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Only keep the logits for the completion tokens if specified
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
            target_ids = input_ids[:, -logits_to_keep:]
        else:
            # Shift for language modeling (predict next token)
            logits = logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
        
        # Convert logits to log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather the log probabilities of the target tokens
        per_token_logps = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        
        return per_token_logps
    
    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep=None):
        """
        Get the last hidden state for text-only models
        Simplified version without multimodal inputs
        """
        # Forward pass to get hidden states
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        
        # Only keep the hidden states for the completion tokens if specified
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]
        
        return last_hidden_state
    
    def compute_reward(self, model_inputs: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """
        Compute rewards for text-only completions
        """
        rewards = []
        
        for reward_func, reward_processing_class in zip(self.reward_funcs, self.reward_processing_classes):
            # Extract prompts and completions for reward computation
            prompts = []
            completions = []
            
            for inputs in model_inputs:
                prompt_text = self.tokenizer.apply_chat_template(
                    inputs["prompt"], 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                completion_text = inputs["completion"]
                
                prompts.append(prompt_text)
                completions.append(completion_text)
            
            # Compute rewards using the reward function
            batch_rewards = reward_func(
                prompts=prompts,
                completions=completions,
                processing_class=reward_processing_class
            )
            
            rewards.append(torch.tensor(batch_rewards, dtype=torch.float32))
        
        # Average rewards if multiple reward functions
        if len(rewards) > 1:
            final_rewards = torch.stack(rewards).mean(dim=0)
        else:
            final_rewards = rewards[0]
        
        return final_rewards
    
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare inputs for text-only training
        """
        # Text-only preparation - no vision processing needed
        prepared_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                prepared_inputs[key] = value.to(self.args.device)
            else:
                prepared_inputs[key] = value
        
        return prepared_inputs
    
    def tokenize_conversation(self, prompt: List[Dict[str, str]], completion: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize conversation for text-only training
        """
        # Apply chat template to the prompt
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize prompt and completion separately
        prompt_tokens = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length
        )
        
        completion_tokens = self.tokenizer(
            completion,
            return_tensors="pt", 
            padding=False,
            truncation=True,
            max_length=self.max_completion_length
        )
        
        return {
            "prompt_ids": prompt_tokens["input_ids"].squeeze(0),
            "prompt_mask": prompt_tokens["attention_mask"].squeeze(0),
            "completion_ids": completion_tokens["input_ids"].squeeze(0),
            "completion_mask": completion_tokens["attention_mask"].squeeze(0),
        }
    
    def _generate_completions(self, prompts: List[Dict[str, str]]) -> List[str]:
        """
        Generate completions for text-only prompts
        """
        completions = []
        
        for prompt in prompts:
            # Apply chat template
            prompt_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_length
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode the completion (remove the prompt part)
            prompt_length = inputs["input_ids"].shape[1]
            completion_ids = outputs[0, prompt_length:]
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            completions.append(completion)
        
        return completions 