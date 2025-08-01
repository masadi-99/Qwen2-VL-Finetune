import os
import torch
from peft import LoraConfig
import ast
import pathlib
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen2ForCausalLM, HfArgumentParser

# Use existing working GRPO trainer and dataset - just modify for text-only
from src.trainer import QwenGRPOTrainer
from src.dataset import make_grpo_data_module
from src.params import DataArguments, ModelArguments, GRPOArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
# NO vision forward patches for text-only
from src.utils import load_reward_funcs

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

# NO configure_vision_tower - text-only
# NO configure_merger - text-only

def configure_llm(model, training_args):
    """Configure LLM parameters for text-only training - simplified from multimodal version"""
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, GRPOArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # NO mixed modality forward for text-only
    training_args.use_liger_loss = False
    
    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
    else:
        training_args.lora_namespan_exclude = []

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    # Load text-only model instead of VL model
    rank0_print(f"Loading text-only model: {model_args.model_id}")
    model = Qwen2ForCausalLM.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
        **bnb_model_from_pretrained_args
    )

    model.config.use_cache = False
    
    # NO vision forward patches for text-only
    
    # Configure only LLM (no vision tower/merger for text-only)
    configure_llm(model, training_args)

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    peft_config = None

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)

    # Use tokenizer instead of processor for text-only
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Use existing working data module - it will handle text-only automatically when no images
    dataset_module = make_grpo_data_module(
        model_id=model_args.model_id,
        processor=tokenizer,  # Pass tokenizer as processor for text-only
        data_args=data_args
    )

    reward_funcs = load_reward_funcs("src.train.reward_funcs")

    # Use existing working trainer
    trainer = QwenGRPOTrainer(
        model=model,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module["eval_dataset"],
        reward_funcs=reward_funcs,
        args=training_args,
        processing_class=tokenizer,  # Use tokenizer for text-only
        peft_config=peft_config,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train() 