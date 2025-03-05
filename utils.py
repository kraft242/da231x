from prompts import minimal_prompt, fluency_prompt
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_from_disk
import torch
from os import path
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

# HELPER FUNCTIONS


def _ensure_gpu_is_available():
    """
    Raises a RuntimeError if torch cannot access the GPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available for training!")


def _get_quantization_config():
    """
    Returns a BitsAndBytesConfig that:
    - Loads in 4 bit mode
    - Quantizes in nf4
    - Uses double quantization
    - Uses bf16 compute type
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _get_model(model_name):
    """
    Creates and prepares a model object for PEFT training by loading a quantization config, enabling gradient checkpointing, and preparing the model for kbit training.
    The prepared model is then returned
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=_get_quantization_config()
    )

    # Prepare model for QLoRa training
    model.gradient_checkpointing_enable()
    return prepare_model_for_kbit_training(model)


def _get_tokenizer(model_name):
    """
    Returns an AutoTokenizer from the provided model_name.
    """
    return AutoTokenizer.from_pretrained(model_name)


def _get_data_collator(tokenizer, model):
    """
    Returns a DataCollatorForSeq2Seq from the provided tokenizer and model.
    """
    return DataCollatorForSeq2Seq(tokenizer, model=model)


def _validate_version(version):
    """
    Ensures the version string is either "minimal" or "fluency", otherwise throws a ValueError.
    """
    if version not in ["minimal", "fluency"]:
        raise ValueError("Unknown dataset version passed.")


def _get_dataset(version):
    """
    Reads the dataset version from disk. The version argument is either "minimal" or "fluency" and the corresponding dataset is returned.
    """
    base_dataset_dir = "datasets"
    dataset_path = path.join(base_dataset_dir, version)
    return load_from_disk(dataset_path)


def _get_prompt(version):
    """
    Returns the corresponding prompt depending on if version is "minimal" or "fluency".
    """
    match version:
        case "minimal":
            return minimal_prompt
        case "fluency":
            return fluency_prompt


def _get_lora_config():
    """
    Returns a LoraConfig object.
    """
    return LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


def _get_training_arguments():
    """
    Returns a TrainingArguments object.
    """
    return TrainingArguments(
        output_dir="tmp_model",
        num_train_epochs=1,
        optim="adamw_bnb_8bit",
        learning_rate=5e-5,
        bf16=True,
        logging_steps=1,
        per_device_train_batch_size=4,
    )

# MAIN FUNCTIONS


def get_model_tokenizer_collator(model_name):
    """
    Creates and returns a model, a tokenizer, and a data-collator object.
    """
    _ensure_gpu_is_available()

    model = _get_model(model_name)
    tokenizer = _get_tokenizer(model_name)
    data_collator = _get_data_collator(tokenizer, model)

    return model, tokenizer, data_collator


def get_tokenized_dataset(tokenizer, version):
    """
    Reads the corresponding dataset version from disk, which is either minimal or fluency.
    The dataset is then processed by prepending the corresponding prompt to each source text and tokenizing all text.
    The dataset is finally returned.
    """
    _validate_version(version)

    dataset = _get_dataset(version)

    prompt = _get_prompt(version)

    def preprocess_function(examples):
        inputs = [prompt + example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        return tokenizer(
            inputs,
            text_target=targets,
            max_length=4096,
            padding="max_length"
        )

    return dataset.map(preprocess_function, batched=True)


def get_trainer(model, tokenizer, tokenized_dataset, data_collator):
    lora_config = _get_lora_config()

    peft_model = get_peft_model(model, lora_config)

    peft_model.config.use_cache = False

    training_arguments = _get_training_arguments()

    return Trainer(
        model=peft_model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator
    )
