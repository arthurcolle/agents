import modal
import argparse
import os
from llama4_finetune import app, create_synthetic_dataset, finetune_llama4_scout, generate_with_finetuned_model, push_model_to_hub

def parse_args():
    parser = argparse.ArgumentParser(description="Run Llama 4 Scout fine-tuning with Unsloth on Modal")
    
    # Main operation modes
    parser.add_argument("--create-dataset", action="store_true", help="Create a synthetic dataset")
    parser.add_argument("--finetune", action="store_true", help="Run fine-tuning")
    parser.add_argument("--generate", action="store_true", help="Generate with fine-tuned model")
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to Hugging Face Hub")
    
    # Dataset creation arguments
    parser.add_argument("--num-examples", type=int, default=100, help="Number of examples for synthetic dataset")
    parser.add_argument("--dataset-path", type=str, default="function_calling_dataset.json", help="Path to dataset (relative to volume)")
    
    # Fine-tuning arguments
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct", help="Base model to fine-tune")
    parser.add_argument("--output-dir", type=str, default="llama4-scout-function-calling", help="Output directory (relative to volume)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, default="How can I check if a number is prime?", help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for generation")
    
    # HF Hub arguments
    parser.add_argument("--repo-id", type=str, help="Hugging Face Hub repository ID")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure HF token is set
    if (args.finetune or args.push_to_hub) and not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN environment variable not set. Set it with: export HF_TOKEN=your_token")
        if args.base_model.startswith("meta-llama/"):
            print("Access to meta-llama models requires an access token!")
            return
    
    # Create dataset
    if args.create_dataset:
        print(f"Creating synthetic dataset with {args.num_examples} examples...")
        result = create_synthetic_dataset.remote(
            num_examples=args.num_examples,
            output_path=f"/datasets/{args.dataset_path}"
        )
        print(f"Dataset created: {result}")
    
    # Run fine-tuning
    if args.finetune:
        print(f"Fine-tuning {args.base_model}...")
        
        # Set up training arguments
        training_args = {
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "optim": "adamw_8bit",
            "logging_steps": 10,
            "save_strategy": "epoch",
            "learning_rate": args.lr,
            "fp16": True,
            "max_grad_norm": 0.3,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "seed": 42,
        }
        
        # Set up LoRA config
        lora_config = {
            "r": args.lora_rank,
            "lora_alpha": args.lora_rank * 2,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
        
        result = finetune_llama4_scout.remote(
            dataset_path=args.dataset_path,
            base_model=args.base_model,
            output_dir=f"/outputs/{args.output_dir}",
            training_args=training_args,
            lora_config=lora_config
        )
        print(f"Fine-tuning completed: {result}")
    
    # Generate with fine-tuned model
    if args.generate:
        print(f"Generating with prompt: {args.prompt}")
        result = generate_with_finetuned_model.remote(
            prompt=args.prompt,
            model_path=f"/outputs/{args.output_dir}",
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"Generated response: {result['response']}")
    
    # Push to Hugging Face Hub
    if args.push_to_hub:
        if not args.repo_id:
            print("Error: --repo-id must be specified for --push-to-hub")
            return
        
        print(f"Pushing model to Hugging Face Hub: {args.repo_id}")
        result = push_model_to_hub.remote(
            model_path=f"/outputs/{args.output_dir}",
            repo_id=args.repo_id,
            commit_message=f"Upload Llama 4 Scout fine-tuned for function calling and reasoning"
        )
        print(f"Push completed: {result}")

if __name__ == "__main__":
    # Run with Modal
    with modal.runner.deploy_stub(app):
        main()