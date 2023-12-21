import os
from datasets import load_dataset
import torch
from torch.utils.tensorboard import SummaryWriter
from trl import SFTTrainer
from evaluate import load
import datetime
from transformers.integrations import TensorBoardCallback

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)

from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightModifierType,
    WeightClipType,
    WeightNoiseType,
    BoundManagementType,
    NoiseManagementType,
    WeightClipParameter,
    WeightModifierParameter,
    MappingParameter,
)

from aihwkit.simulator.presets import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD


# Pretrained LLAMA model from Hugging Face model hub
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
access_token = "..."
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Training parameters
learning_rate = 0.0001


def create_rpu_config(tile_size=512, dac_res=256, adc_res=256):
    """Create RPU Config emulated typical PCM Device"""

    rpu_config = InferenceRPUConfig(
        clip=WeightClipParameter(type=WeightClipType.FIXED_VALUE, fixed_value=1.0),
        modifier=WeightModifierParameter(
            rel_to_actual_wmax=True, type=WeightModifierType.ADD_NORMAL),
        
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=True,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(
            w_noise_type=WeightNoiseType.PCM_READ,
            w_noise=0.0175,
            inp_res=dac_res,
            out_res=adc_res,
            out_bound=10.0,
            out_noise=0.04,
            bound_management=BoundManagementType.ITERATIVE,
            noise_management=NoiseManagementType.ABS_MAX,
        ),
        noise_model=PCMLikeNoiseModel(),
        drift_compensation=GlobalDriftCompensation(),
    )
    return rpu_config


def create_model(rpu_config):
    """Return pre-trained model and whether or not it was loaded from a checkpoint"""

    model = AutoModel.from_pretrained(MODEL_NAME)

    model = convert_to_analog(model, rpu_config)
    model.remap_analog_weights()

    print(model)
    return model


def make_trainer(model, optimizer, tokenized_data):
    """Create the Huggingface Trainer"""
    training_args = TrainingArguments(
        output_dir="./",
        save_strategy="no",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.001,
        no_cuda=False,
    )

    collator = DefaultDataCollator()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_data,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        callbacks=[TensorBoardCallback(writer)],
    )

    return trainer, writer


def main():
    
    rpu_config = create_rpu_config()
    
    model = create_model(rpu_config)

    dataset = load_dataset("mlabonne/guanaco-llama2-1k")
    tokenized_dataset = tokenizer(dataset["text"], padding=True, truncation=True, max_length=100)

    """Create the analog-aware optimizer"""
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(model)

    trainer, writer = make_trainer(model, optimizer, tokenized_dataset)

    # Train the model
    trainer.train()
    torch.save(model.state_dict(), "model.pt")

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(eval_results)


if __name__ == "__main__":
    main()
