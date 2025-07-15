# CoE: A Clue of Emotion Framework for Emotion Recognition in Conversations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Zh1yuShen%2FCoE-blue.svg)](https://github.com/Zh1yuShen/CoE)

This repository contains the official implementation of **"CoE: A Clue of Emotion Framework for Emotion Recognition in Conversations"**.

**📄 Paper**: The paper will be available soon on the ACL official website.

## Abstract

Emotion Recognition in Conversations (ERC) is crucial for machines to understand dynamic human emotions. While Large Language Models (LLMs) show promise, their performance is often limited by challenges in interpreting complex conversational streams. We introduce a **Clue of Emotion (CoE)** framework, which progressively integrates key conversational clues to enhance the ERC task. Building on CoE, we implement a multi-stage auxiliary learning strategy that incorporates role-playing, speaker identification, and emotion reasoning tasks.

## Framework Overview

The CoE framework consists of three main components:

### Auxiliary Tasks
1. **Role-Playing Task**: Enhances understanding of character personalities and conversational dynamics
2. **Speaker Identification Task**: Improves speaker-aware emotion modeling by learning to identify speakers
3. **STEeR (Self-Taught Emotion Reasoning)**: Employs iterative rationale generation to improve emotional reasoning

### Textual Clues Integration
- **Persona Information**: Character personalities and backgrounds
- **Scene Context**: Environmental and situational information  
- **Dialogue History**: Previous conversational context

### Multi-Stage Training Strategy
- Progressive training through auxiliary tasks
- Mixed auxiliary learning for optimal performance
- Final fine-tuning on ERC task

## Training Strategy

### Curriculum Learning Approach

The CoE framework implements a curriculum learning strategy where models progressively learn from simple to complex tasks:

**Stage 1: Auxiliary Tasks (Optional)**
- **Role-Playing Task**: Learn character personalities and dialogue patterns
- **Speaker Identification Task**: Learn to distinguish between different speakers

**Stage 2: STEeR (Self-Taught Emotion Reasoning)**
- Generate emotion reasoning explanations for training examples
- Dynamically expand the training set with high-quality reasoned examples
- Learn to provide rationales before making emotion predictions

**Stage 3: Final ERC Training**
- Fine-tune on the main emotion recognition task
- Leverage knowledge from previous stages through checkpoint initialization

### STEeR Data Generation Process

STEeR (Self-Taught Emotion Reasoning) implements a dynamic learning mechanism:

1. **Initial Training Set**: Start with `trainbegin.json` containing 10 examples with complete emotional reasoning rationales

2. **Inference Phase**: During each training epoch, the model generates predictions and rationales for examples in `train4infer.json`

3. **Quality Filtering**: Only examples where the model correctly predicts emotions are added to the expanding training set

4. **Dynamic Updates**: 
   - `memory.json`: Continuously grows with correctly reasoned examples
   - `train4infer.json`: Updated with remaining examples for next iteration

5. **Distributed Collection**: In multi-GPU training, rationales from all GPUs are collected and aggregated to ensure comprehensive data coverage

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU(s)
- DeepSpeed for distributed training

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Zh1yuShen/CoE.git
cd CoE
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Configuration

1. **Download a base model** (e.g., Mistral-7B-v0.1 from Hugging Face)

2. **Update model and data paths** in `scripts/train_coe.sh`:
```bash
MODEL_PATH='/path/to/your/model'
DATA_PATH='/path/to/your/dataset'
```

3. **Configure your GPUs**:
```bash
export CUDA_VISIBLE_DEVICES="0,1"  # Set your available GPU IDs
```

## Usage

### Training Strategies

#### Default: Full Curriculum Learning (Sequential Training)
```bash
cd scripts
# Edit train_coe.sh and set:
TRAINING_STRATEGY="full_curriculum"

bash train_coe.sh
```

This approach trains tasks sequentially: Role-Playing → Speaker ID → STEeR → Final ERC.

#### STEeR-Only Training
```bash
# Edit train_coe.sh and set:
TRAINING_STRATEGY="steer_only"

bash train_coe.sh
```

#### Auxiliary Tasks Only
```bash
# Edit train_coe.sh and set:
TRAINING_STRATEGY="auxiliary_only"
```

## Advanced Strategy Implementation

Based on our paper findings, **Strategy B (Mixed Auxiliary with Pre-generated Rationales)** achieved the best performance (EmoryNLP: 44.29%, MELD: 70.11%). Here's how to implement different strategies manually:

### Strategy A: Sequential Curriculum Training (Default Implementation)
This is what the script implements by default - sequential training of each task.

### Strategy B: Mixed Auxiliary with Pre-generated Rationales (Best Performance)

To implement the paper's best-performing strategy, follow these steps:

#### Step 1: Pre-generate STEeR Rationale Dataset
```bash
# First, run STEeR-only training to generate rationales
TRAINING_STRATEGY="steer_only"
bash train_coe.sh

# This will create an enriched dataset with rationales in:
# ./experiments/[MODEL]/lora/[DATASET]/STEeR/memory.json
```

#### Step 2: Create Mixed Training Dataset
```bash
# Combine datasets for joint training
# You can create a mixed dataset by concatenating:
# - Role-playing task data
# - Speaker identification task data  
# - STEeR-generated rationale data

# Example script to mix datasets:
cat /path/to/roleplay_data.json \
    /path/to/speaker_data.json \
    ./experiments/[MODEL]/lora/[DATASET]/STEeR/memory.json \
    > mixed_auxiliary_dataset.json
```

#### Step 3: Joint Training on Mixed Dataset
```bash
# Manually train on the mixed dataset using erc_trainer.py
deepspeed --master_port=2236 ../src/erc_trainer.py \
    --dataset "emorynlp" \
    --model_name_or_path /path/to/your/model \
    --data_dir /path/to/mixed_auxiliary_dataset \
    --output_dir ./experiments/mixed_auxiliary_training \
    --max_length 1400 \
    --batch_size 12 \
    --deepspeed_config ../config/deepspeed_config.json \
    --num_train_epochs 5 \
    --lora True \
    --learning_rate 2e-4 \
    --do_train True \
    --do_eval True \
    --statistic_mode False
```

#### Step 4: Final ERC Training
```bash
# Use the mixed auxiliary trained model for final ERC training
deepspeed --master_port=2236 ../src/erc_trainer.py \
    --dataset "emorynlp" \
    --model_name_or_path /path/to/your/base/model \
    --data_dir /path/to/main/erc/dataset \
    --output_dir ./experiments/final_erc \
    --checkpoint_dir ./experiments/mixed_auxiliary_training \
    --max_length 1400 \
    --batch_size 12 \
    --num_train_epochs 20 \
    --lora True \
    --learning_rate 2e-4 \
    --do_train True \
    --do_eval True \
    --statistic_mode True
```

### Strategy Implementation Tips

1. **Data Mixing**: For joint training, ensure consistent data format across all auxiliary tasks
2. **Rationale Quality**: Monitor the quality of STEeR-generated rationales before mixing
3. **Batch Balancing**: When mixing datasets, consider balancing the number of examples from each task
4. **Learning Rate Adjustment**: Joint training may require different learning rates than sequential training
5. **Evaluation Strategy**: Regularly evaluate on validation sets during joint training to prevent overfitting

### Training Process Explanation

**Role-Playing Task Training**:
- Trains the model to generate character-appropriate responses
- Uses persona information and dialogue context
- Improves understanding of character dynamics

**Speaker Identification Task Training**:
- Learns to identify speakers based on dialogue content and personas
- Enhances speaker-aware emotion modeling capabilities
- Builds foundational understanding of conversational structure

**STEeR Task Training**:
- Starts with a small set of examples with reasoning explanations
- Iteratively generates rationales and expands the training set
- Implements quality control by only keeping correctly predicted examples
- Enables the model to learn explicit reasoning patterns

**Final ERC Training**:
- Uses knowledge from all previous stages
- Focuses on emotion classification performance
- Leverages enhanced understanding from auxiliary tasks

### Evaluation

Evaluation is performed automatically during training after each epoch. Results are saved in the model output directory with files like `*_preds_for_eval_*.text` containing detailed prediction results and metrics.

## Supported Datasets

- **EmoryNLP**: Emotion recognition on TV show Friends
- **MELD**: Multimodal emotion recognition dataset  
- **IEMOCAP**: Interactive emotional dyadic motion capture database

## Project Structure

```
CoE/
├── src/                          # Source code
│   ├── steer_trainer.py         # STEeR task training
│   ├── erc_trainer.py           # ERC and auxiliary tasks training
│   └── data_utils/              # Data processing utilities
├── scripts/                     # Training scripts
│   └── train_coe.sh            # Main training script
├── config/                      # Configuration files
└── docs/                        # Documentation
```

## Framework Details

### Training Method
- **LoRA (Low-Rank Adaptation)** fine-tuning for efficient training
- **DeepSpeed ZeRO** for memory optimization
- **Multi-GPU support** with distributed training

### Model Support
- Primary model: **Mistral-7B-v0.1**
- Compatible with other instruction-tuned models

### Data Format
The framework uses structured prompts with:
- Speaker personas and character information
- Scene context and environmental details
- Dialogue history for conversational context

**Required Data Files for STEeR Task**:
- `trainbegin.json`: Initial examples with reasoning rationales
- `train.json`: Main training examples (emotion labels only)
- `test.json`: Test set for evaluation

## Advanced Usage

### Custom Dataset Integration

1. **Prepare your dataset** in the required JSON format with the following structure:
   - Each line contains one training example
   - Include input text with personas, scene, and dialogue
   - Provide emotion labels for training

2. **Add dataset configuration** in `train_coe.sh`:
   ```bash
   'your_dataset')
       MAX_LENGTH=1400
       DATA_PATH='/path/to/your/dataset'
       DATA_PATH_LEARNER='/path/to/your/steer_data'
       ;;
   ```

3. **Update data paths** and run training

### Multi-GPU Training

Configure your GPU setup:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Use available GPUs
CONFIG_NUM_GPUS=4                       # Set number of GPUs
```

### Training Tips

1. **Start with STEeR**: If you're unsure which strategy to use, begin with STEeR-only training as it provides the most direct benefit to emotion recognition.

2. **Monitor Data Growth**: During STEeR training, check the growth of `memory.json` to ensure the model is successfully generating quality rationales.

3. **Curriculum Learning**: For best results, use the full curriculum approach, especially on complex datasets with rich character interactions.

4. **Checkpoint Management**: The framework automatically uses checkpoints from previous stages, ensuring knowledge transfer between tasks.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This implementation is for research purposes. Please ensure you have the proper permissions and licenses for the models and datasets you use. 