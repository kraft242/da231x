from prompts import minimal_prompt, fluency_prompt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_from_disk
import torch
from os import path, makedirs
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# HELPER FUNCTIONS


def _ensure_gpu_is_available():
    """
    Raises a RuntimeError if torch cannot access the GPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available for training!")


def _validate_version(version):
    """
    Ensures the version string is either "minimal" or "fluency", otherwise throws a ValueError.
    """
    if version not in ["minimal", "fluency"]:
        raise ValueError("Unknown dataset version passed.")


def _get_prompt(version):
    """
    Returns the corresponding prompt depending on if version is "minimal" or "fluency".
    """
    match version:
        case "minimal":
            return minimal_prompt
        case "fluency":
            return fluency_prompt


def _get_model_path(version):
    """
    Ensures that the directory ./models/version exists and returns the path.
    The version argument is either "minimal" or "fluency"
    """

    full_model_dir_path = path.join("models", version)
    makedirs(full_model_dir_path, exist_ok=True)
    return full_model_dir_path


# MAIN FUNCTIONS


def get_model_tokenizer_collator(model_name):
    """
    Creates and returns a model, a tokenizer, and a data-collator object.
    """
    _ensure_gpu_is_available()

    # Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    return model, tokenizer, data_collator


def get_tokenized_dataset(tokenizer, version):
    """
    Reads the corresponding dataset version from disk, which is either minimal or fluency.
    The dataset is then processed by prepending the corresponding prompt to each source text and tokenizing all text.
    The dataset is finally returned.
    """
    _validate_version(version)

    # Non-Tokenized Dataset
    base_dataset_dir = "datasets"
    dataset_path = path.join(base_dataset_dir, version)
    dataset = load_from_disk(dataset_path)

    # Prompt
    prompt = _get_prompt(version)

    def preprocess_function(examples):
        inputs = [prompt + example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        return tokenizer(
            inputs, text_target=targets, max_length=4096, padding="max_length"
        )

    return dataset.map(preprocess_function, batched=True)


def get_trainer(model, tokenizer, tokenized_dataset, data_collator, version):
    # Make model PEFT compatible
    lora_config = LoraConfig(r=128, lora_alpha=64, bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, lora_config)
    peft_model.config.use_cache = False

    # Training Arguments
    epochs = 3
    batch_size = 4
    dataset_size = len(tokenized_dataset["train"])
    steps_per_epoch = (dataset_size * epochs) // batch_size
    num_eval_steps = steps_per_epoch // 2

    training_arguments = TrainingArguments(
        output_dir=_get_model_path(version),
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=num_eval_steps,
        prediction_loss_only=True,
        optim="adamw_bnb_8bit",
        learning_rate=5e-5,
        bf16=True,
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        save_strategy="epoch",
        overwrite_output_dir=True,
    )

    return Trainer(
        model=peft_model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
