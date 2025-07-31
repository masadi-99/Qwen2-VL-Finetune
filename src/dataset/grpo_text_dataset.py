import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import TextDataArguments
from src.constants import SYSTEM_MESSAGE


def llava_to_openai_text(conversations):
    """Convert LLaVA format to OpenAI format for text-only conversations"""
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        # Remove any image tokens that might be present
        content = conversation["value"].replace("<image>", "").replace("<video>", "").strip()
        
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


class GRPOTextDataset(Dataset):
    """Dataset for GRPO training with text-only data"""

    def __init__(
        self,
        data_path: str | list,
        tokenizer: transformers.PreTrainedTokenizerBase,
        data_args: TextDataArguments,
        model_id,
        padding=True,
    ):
        super(GRPOTextDataset, self).__init__()
        
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.max_length = data_args.max_seq_length

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        # Convert conversations to text-only format
        conversations = copy.deepcopy(llava_to_openai_text(sources['conversations']))

        # Extract user prompt and assistant response
        user_input = conversations[0]
        gpt_response = conversations[1]

        # Build the prompt for text-only training
        user_prompt = [{"role": "user", "content": user_input['content']}]

        # Add system message if present
        if len(SYSTEM_MESSAGE) > 0:
            system_message = {"role": "system", "content": SYSTEM_MESSAGE}
            user_prompt.insert(0, system_message)
        
        data_dict = dict(
            prompt=user_prompt,
            assistant=gpt_response,
        )

        return data_dict


def make_grpo_text_data_module(model_id, tokenizer, data_args):
    """Make dataset and collator for text-only GRPO training."""
    
    train_dataset = GRPOTextDataset(
        data_path=data_args.data_path, 
        tokenizer=tokenizer, 
        data_args=data_args, 
        model_id=model_id
    )

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = GRPOTextDataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            model_id=model_id
        )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    ) 