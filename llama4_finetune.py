import os
import modal
import json
import torch
from datasets import load_dataset
from typing import List, Dict, Any

# Define the Modal image with all necessary dependencies
image = modal.Image.debian_slim().pip_install(
    "unsloth",
    "transformers>=4.37.2",
    "datasets",
    "bitsandbytes>=0.41.1",
    "accelerate>=0.25.0",
    "scipy",
    "einops",
    "wandb",
    "peft>=0.5.0",
    "trl>=0.7.4",
    "huggingface_hub",
    "sentencepiece",
)

# Define Modal volumes for persistent storage
models_volume = modal.Volume.from_name("llama4-models", create_if_missing=True)
outputs_volume = modal.Volume.from_name("llama4-outputs", create_if_missing=True)
datasets_volume = modal.Volume.from_name("llama4-datasets", create_if_missing=True)

# Create Modal app
app = modal.App("llama4-finetune")

# System prompt for function calling and reasoning
SYSTEM_PROMPT = """You are an expert assistant with advanced reasoning capabilities. You're skilled at using tools through function calling to accomplish complex tasks. You understand when to use functions vs when to solve problems directly.

When presented with a problem:
1. Carefully break down the problem into logical steps
2. Identify if using external tools would help solve the problem
3. For function calls, precisely follow JSON schema format
4. When reasoning, explain your thought process clearly step by step

Always ensure your function call parameters match the required schema exactly.
"""

# Sample function calling dataset format
FUNCTION_SCHEMA = {
    "name": "search_knowledge_base",
    "description": "Search for information in a knowledge base",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "filters": {
                "type": "object",
                "description": "Optional filters to apply",
                "properties": {
                    "date_range": {"type": "string", "description": "Time period for results"},
                    "category": {"type": "string", "description": "Category to filter by"}
                }
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return"
            }
        },
        "required": ["query"]
    }
}

# Sample prompt formatting functions
def create_function_calling_prompt(user_query, available_functions):
    """Format a prompt for function calling training"""
    function_descriptions = json.dumps(available_functions, indent=2)
    
    return f"""<|header_start|>system<|header_end|>
{SYSTEM_PROMPT}

Available functions:
{function_descriptions}
<|eot|>
<|header_start|>user<|header_end|>
{user_query}
<|eot|>
<|header_start|>assistant<|header_end|>"""

def create_reasoning_prompt(user_query):
    """Format a prompt for reasoning training"""
    return f"""<|header_start|>system<|header_end|>
{SYSTEM_PROMPT}
<|eot|>
<|header_start|>user<|header_end|>
{user_query}
<|eot|>
<|header_start|>assistant<|header_end|>"""

# Function to process and format the dataset
def format_function_calling_dataset(dataset):
    """Process and format the dataset for function calling training"""
    formatted_data = []
    
    for item in dataset:
        # Extract relevant fields (adjust according to your dataset structure)
        user_query = item["input"]
        model_response = item["output"]
        available_functions = item.get("available_functions", [FUNCTION_SCHEMA])
        
        # Create formatted prompt
        prompt = create_function_calling_prompt(user_query, available_functions)
        
        # Create formatted example
        formatted_data.append({
            "text": f"{prompt}\n{model_response}<|eot|>"
        })
    
    return formatted_data

def format_reasoning_dataset(dataset):
    """Process and format the dataset for reasoning training"""
    formatted_data = []
    
    for item in dataset:
        # Extract relevant fields
        user_query = item["input"]
        model_response = item["output"]
        
        # Create formatted prompt
        prompt = create_reasoning_prompt(user_query)
        
        # Create formatted example
        formatted_data.append({
            "text": f"{prompt}\n{model_response}<|eot|>"
        })
    
    return formatted_data

@app.function(
    gpu="A100",
    timeout=86400,
    volumes={
        "/models": models_volume,
        "/outputs": outputs_volume,
        "/datasets": datasets_volume,
    },
    memory=80000,
)
def finetune_llama4_scout(
    dataset_path: str = "synthetic_function_calling_dataset.json",
    base_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    output_dir: str = "/outputs/llama4-scout-function-calling",
    training_args: Dict[str, Any] = None,
    lora_config: Dict[str, Any] = None,
):
    """
    Fine-tune Llama 4 Scout model for function calling and reasoning using QLoRA.
    
    Args:
        dataset_path: Path to the dataset JSON file
        base_model: Base model to fine-tune
        output_dir: Directory to save the fine-tuned model
        training_args: Training arguments
        lora_config: LoRA configuration
    """
    import json
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    
    print(f"Starting fine-tuning of {base_model}")
    print(f"Loading dataset from {dataset_path}")
    
    # Set default training arguments if not provided
    if training_args is None:
        training_args = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "optim": "adamw_8bit",
            "logging_steps": 10,
            "save_strategy": "epoch",
            "learning_rate": 2e-5,
            "fp16": True,
            "max_grad_norm": 0.3,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "seed": 42,
        }
    
    # Set default LoRA config if not provided
    if lora_config is None:
        lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
    
    # Load the dataset
    dataset_path = os.path.join("/datasets", dataset_path)
    if not os.path.exists(dataset_path):
        # If dataset doesn't exist, let's create a small synthetic one for testing
        print(f"Dataset not found at {dataset_path}, creating a synthetic dataset...")
        
        synthetic_data = []
        for i in range(10):
            synthetic_data.append({
                "input": f"What is the capital of France? Search for detailed information.",
                "output": """I'll help you find information about the capital of France.

I should use the search_knowledge_base function for this.

```json
{
  "function": "search_knowledge_base", 
  "parameters": {
    "query": "capital of France",
    "filters": {
      "category": "geography"
    },
    "limit": 5
  }
}
```

Let me analyze the results when they return.""",
                "available_functions": [FUNCTION_SCHEMA]
            })
        
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, 'w') as f:
            json.dump(synthetic_data, f)
        
        print(f"Created synthetic dataset with {len(synthetic_data)} examples")
    
    # Load and process the dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Format the dataset
    formatted_data = format_function_calling_dataset(dataset)
    
    # Load the model with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        token=os.environ.get("HF_TOKEN"),
        # Use unsloth's optimized 4-bit quantization
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
        use_gradient_checkpointing=training_args["gradient_checkpointing"],
        random_seed=training_args["seed"],
        use_rslora=False,  # Regular LoRA, not rank-stabilized
    )
    
    # Set up training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_args["num_train_epochs"],
        per_device_train_batch_size=training_args["per_device_train_batch_size"],
        gradient_accumulation_steps=training_args["gradient_accumulation_steps"],
        gradient_checkpointing=training_args["gradient_checkpointing"],
        optim=training_args["optim"],
        logging_steps=training_args["logging_steps"],
        save_strategy=training_args["save_strategy"],
        learning_rate=training_args["learning_rate"],
        fp16=training_args["fp16"],
        max_grad_norm=training_args["max_grad_norm"],
        warmup_ratio=training_args["warmup_ratio"],
        lr_scheduler_type=training_args["lr_scheduler_type"],
        seed=training_args["seed"],
        report_to=["tensorboard"],
    )
    
    # Create dataset
    train_dataset = {"text": [item["text"] for item in formatted_data]}
    
    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        args=args,
        packing=True,
        max_seq_length=4096,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    print(f"Training complete. Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    # Merge and save weights if needed
    if os.environ.get("SAVE_MERGED", "false").lower() == "true":
        print("Merging and saving weights...")
        FastLanguageModel.save_pretrained(
            model, tokenizer, output_dir + "-merged",
            save_method="merged",
        )
    
    print("Fine-tuning completed successfully!")
    return {"status": "success", "output_dir": output_dir}

@app.function(
    gpu="A100",
    timeout=3600,
    volumes={
        "/models": models_volume,
        "/outputs": outputs_volume,
    },
    memory=24000,
)
def generate_with_finetuned_model(
    prompt: str,
    model_path: str = "/outputs/llama4-scout-function-calling",
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
    min_p: float = 0.01,
):
    """
    Generate text using the fine-tuned model.
    
    Args:
        prompt: The prompt to generate from
        model_path: Path to the fine-tuned model
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p sampling parameter
        min_p: Min-p sampling parameter
    """
    import torch
    from unsloth import FastLanguageModel
    from transformers import GenerationConfig
    
    print(f"Loading model from {model_path}")
    
    # Load the fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    # Format the prompt
    formatted_prompt = f"""<|header_start|>system<|header_end|>
{SYSTEM_PROMPT}
<|eot|>
<|header_start|>user<|header_end|>
{prompt}
<|eot|>
<|header_start|>assistant<|header_end|>"""
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    
    # Decode the outputs
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the assistant's response
    response = generated_text.split("<|header_start|>assistant<|header_end|>")[-1].strip()
    
    print(f"Generated response: {response}")
    return {"response": response}

@app.function(
    gpu="A100",
    timeout=3600,
    volumes={"/outputs": outputs_volume},
    memory=24000,
)
def push_model_to_hub(
    model_path: str = "/outputs/llama4-scout-function-calling",
    repo_id: str = None,
    commit_message: str = "Upload Llama 4 Scout fine-tuned for function calling",
):
    """
    Push the fine-tuned model to the Hugging Face Hub.
    
    Args:
        model_path: Path to the fine-tuned model
        repo_id: Hugging Face Hub repository ID
        commit_message: Commit message
    """
    from huggingface_hub import HfApi
    import os
    
    if not repo_id:
        raise ValueError("repo_id must be provided")
    
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN environment variable must be set")
    
    print(f"Pushing model from {model_path} to {repo_id}")
    
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, exist_ok=True, private=True)
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    
    print(f"Successfully pushed model to {repo_id}")
    return {"status": "success", "repo_id": repo_id}

@app.function()
def create_synthetic_dataset(
    num_examples: int = 100,
    output_path: str = "/datasets/function_calling_dataset.json",
):
    """
    Create a synthetic dataset for fine-tuning.
    
    Args:
        num_examples: Number of examples to generate
        output_path: Path to save the dataset
    """
    import json
    import os
    import random
    
    # Define function schemas
    function_schemas = [
        {
            "name": "search_knowledge_base",
            "description": "Search for information in a knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters to apply",
                        "properties": {
                            "date_range": {"type": "string", "description": "Time period for results"},
                            "category": {"type": "string", "description": "Category to filter by"}
                        }
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "calculate_statistics",
            "description": "Calculate statistics on a numerical dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array", 
                        "items": {"type": "number"},
                        "description": "Array of numeric values"
                    },
                    "operations": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["mean", "median", "sum", "std", "min", "max"]},
                        "description": "Statistical operations to perform"
                    }
                },
                "required": ["data", "operations"]
            }
        }
    ]
    
    # Sample queries for function calling
    function_call_templates = [
        {"query": "What is the capital of {country}?", "function": "search_knowledge_base"},
        {"query": "Find information about {topic} in {field}", "function": "search_knowledge_base"},
        {"query": "Calculate the average and standard deviation of these numbers: {numbers}", "function": "calculate_statistics"},
        {"query": "What's the mean and median of this dataset: {numbers}", "function": "calculate_statistics"},
        {"query": "I need to understand {complex_topic}. Can you help me research it?", "function": "search_knowledge_base"},
        {"query": "Find the minimum and maximum values in this data: {numbers}", "function": "calculate_statistics"},
    ]
    
    # Sample data for substitution
    countries = ["France", "Japan", "Brazil", "Egypt", "Australia", "Canada", "India", "South Africa"]
    topics = ["quantum computing", "climate change", "artificial intelligence", "renewable energy", "blockchain", "machine learning"]
    fields = ["science", "technology", "economics", "history", "politics", "medicine"]
    complex_topics = [
        "how neural networks process information", 
        "the impact of rising sea levels on coastal cities",
        "how mRNA vaccines work",
        "the ethical implications of autonomous vehicles",
        "distributed systems architecture"
    ]
    
    # Create dataset
    dataset = []
    for i in range(num_examples):
        # Select a random template
        template = random.choice(function_call_templates)
        query = template["query"]
        function_name = template["function"]
        
        # Get the corresponding function schema
        function_schema = next(schema for schema in function_schemas if schema["name"] == function_name)
        
        # Fill in the template
        if "{country}" in query:
            query = query.replace("{country}", random.choice(countries))
        if "{topic}" in query:
            query = query.replace("{topic}", random.choice(topics))
        if "{field}" in query:
            query = query.replace("{field}", random.choice(fields))
        if "{complex_topic}" in query:
            query = query.replace("{complex_topic}", random.choice(complex_topics))
        if "{numbers}" in query:
            numbers = [round(random.uniform(1, 100), 2) for _ in range(random.randint(5, 10))]
            query = query.replace("{numbers}", ", ".join(map(str, numbers)))
        
        # Create the response based on the function
        if function_name == "search_knowledge_base":
            # Create a function call for knowledge base search
            response = f"""I'll help you find information about this.

I should use the search_knowledge_base function to look this up.

```json
{{
  "function": "search_knowledge_base", 
  "parameters": {{
    "query": "{query.split('?')[0] if '?' in query else query}",
    "filters": {{
      "category": "{random.choice(fields)}"
    }},
    "limit": {random.randint(3, 10)}
  }}
}}
```

After searching the knowledge base, I can provide you with the detailed information you need."""
        
        elif function_name == "calculate_statistics":
            # Parse the numbers from the query
            numbers_str = query.split(":")[1].strip() if ":" in query else "5, 10, 15, 20, 25"
            numbers = [float(num.strip()) for num in numbers_str.split(",") if num.strip()]
            
            # Determine which operations to perform based on the query
            operations = []
            if "mean" in query or "average" in query:
                operations.append("mean")
            if "median" in query:
                operations.append("median")
            if "standard deviation" in query or "std" in query:
                operations.append("std")
            if "minimum" in query or "min" in query:
                operations.append("min")
            if "maximum" in query or "max" in query:
                operations.append("max")
            
            # If no specific operations mentioned, include all
            if not operations:
                operations = ["mean", "median", "std", "min", "max"]
            
            # Create a function call for calculating statistics
            response = f"""I'll help you calculate statistics on this data.

I'll use the calculate_statistics function for this.

```json
{{
  "function": "calculate_statistics",
  "parameters": {{
    "data": {numbers},
    "operations": {operations}
  }}
}}
```

Once I have the results, I can provide you with the calculated statistics you requested."""
        
        # Add to dataset
        dataset.append({
            "input": query,
            "output": response,
            "available_functions": [function_schema]
        })
    
    # Save the dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created synthetic dataset with {num_examples} examples at {output_path}")
    return {"status": "success", "path": output_path, "num_examples": num_examples}

if __name__ == "__main__":
    # For local testing with Modal
    modal.runner.deploy_stub(app)