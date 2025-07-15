#!/bin/bash

# ==============================================================================
# CoE (Clue of Emotion) Training Script
# 
# This script implements the CoE framework for Emotion Recognition in Conversations
# with support for auxiliary tasks: Role-Playing, Speaker Identification, and STEeR
# ==============================================================================

# ==============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS AS NEEDED
# ==============================================================================

# GPU Configuration
export CUDA_VISIBLE_DEVICES="0,1"  # Set your available GPU IDs
CONFIG_NUM_GPUS=2                   # Number of GPUs to use

# Model Configuration
MODEL_NAME='Mistral'                # Options: 'Mistral', 'LLaMA3', etc.

# Dataset Configuration
DATASET='emorynlp'                  # Options: 'emorynlp', 'meld', 'iemocap'

# Training Configuration (LoRA only)
BATCH_SIZE_PER_GPU=3
ACCUMULATION_STEPS=2
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * CONFIG_NUM_GPUS * ACCUMULATION_STEPS))
LEARNING_RATE=2e-4

# ==============================================================================
# TRAINING STRATEGY CONFIGURATION
# ==============================================================================

# Choose one training approach:

# Full Curriculum Learning (Recommended)
# Sequential training: Role-Playing -> Speaker ID -> STEeR -> Final ERC
TRAINING_STRATEGY="full_curriculum"

# STEeR-Only Training  
# Focuses solely on emotion reasoning without auxiliary tasks
# TRAINING_STRATEGY="steer_only"

# Auxiliary Tasks Only
# Trains only role-playing and speaker identification tasks
# TRAINING_STRATEGY="auxiliary_only"

# Custom (Manual configuration)
# Set individual task switches below
# TRAINING_STRATEGY="custom"

# ==============================================================================
# INDIVIDUAL TASK CONFIGURATION (for custom strategy)
# ==============================================================================

# Task switches - Only used when TRAINING_STRATEGY="custom"
# Set to 'True' to enable, 'False' to disable
# Training order: Role-Playing -> Speaker -> STEeR -> Final ERC

ROLE_PLAYING_TASK='False'           # Role-playing auxiliary task (character dynamics)
SPEAKER_IDENTIFICATION_TASK='False' # Speaker identification auxiliary task (speaker awareness)  
STEER_TASK='False'                  # STEeR (Self-Taught Emotion Reasoning) task

# Training Epochs for each task
ERC_EPOCHS=20                       # Main ERC task epochs
ROLE_PLAYING_EPOCHS=2              # Role-playing task epochs  
SPEAKER_EPOCHS=2                   # Speaker identification epochs
STEER_EPOCHS=8                     # STEeR task epochs

# ==============================================================================
# AUTOMATIC TASK CONFIGURATION BASED ON STRATEGY
# ==============================================================================

case ${TRAINING_STRATEGY} in
    "full_curriculum")
        echo "Using Full Curriculum Learning (Sequential Training)"
        echo "Training order: Role-Playing -> Speaker ID -> STEeR -> Final ERC"
        ROLE_PLAYING_TASK='True'
        SPEAKER_IDENTIFICATION_TASK='True'  
        STEER_TASK='True'
        ;;
    "steer_only")
        echo "Using STEeR-Only Training"
        ROLE_PLAYING_TASK='False'
        SPEAKER_IDENTIFICATION_TASK='False'  
        STEER_TASK='True'
        ;;
    "auxiliary_only")
        echo "Using Auxiliary Tasks Only"
        ROLE_PLAYING_TASK='True'
        SPEAKER_IDENTIFICATION_TASK='True'  
        STEER_TASK='False'
        ;;
    "custom")
        echo "Using Custom Configuration"
        echo "Role-Playing: ${ROLE_PLAYING_TASK}"
        echo "Speaker ID: ${SPEAKER_IDENTIFICATION_TASK}"
        echo "STEeR: ${STEER_TASK}"
        ;;
    *)
        echo "ERROR: Invalid TRAINING_STRATEGY. Please set one of: full_curriculum, steer_only, auxiliary_only, custom"
        exit 1
        ;;
esac

# Port Configuration (change if port is occupied)
MASTER_PORT=2236

# ==============================================================================
# MODEL PATH CONFIGURATION
# ==============================================================================

case ${MODEL_NAME} in
    'Mistral')
        MODEL_PATH='/path/to/Mistral-7B-v0.1'  # Update with your model path
        ;;
    'LLaMA3')
        MODEL_PATH='/path/to/Llama-3.1-8B-Instruct'  # Update with your model path
        ;;
    *)
        echo "ERROR: Unsupported model ${MODEL_NAME}. Please add model path configuration."
        exit 1
        ;;
esac

# ==============================================================================
# DATASET PATH CONFIGURATION
# ==============================================================================

case ${DATASET} in
    'emorynlp')
        MAX_LENGTH=1400
        DATA_PATH='/path/to/Emorynlp_ERCdatav2'           # Update with your data path
        DATA_PATH_LEARNER='/path/to/Emorynlp_ERC_learnerv2'  # Update with your data path
        DATA_SPEAKER_PATH='/path/to/Emorynlp_speaker'     # Update with your data path
        DATA_ROLEPLAY_PATH='/path/to/Emorynlp_roleplay'   # Update with your data path
        ;;
    'meld')
        MAX_LENGTH=1400
        DATA_PATH='/path/to/MELD_ERCdata'                 # Update with your data path
        DATA_PATH_LEARNER='/path/to/MELD_ERC_learner'     # Update with your data path
        # Add other paths as needed
        ;;
    'iemocap')
        MAX_LENGTH=1400
        DATA_PATH='/path/to/IEMOCAP_ERCdata'              # Update with your data path
        DATA_PATH_LEARNER='/path/to/IEMOCAP_ERC_learner'  # Update with your data path
        # Add other paths as needed
        ;;
    *)
        echo "ERROR: Unsupported dataset ${DATASET}"
        exit 1
        ;;
esac

# ==============================================================================
# PATH VALIDATION
# ==============================================================================

# Check if model path exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path does not exist: ${MODEL_PATH}"
    echo "Please update MODEL_PATH in the script configuration section."
    exit 1
fi

# Check if required data paths exist
if [ ! -d "${DATA_PATH}" ]; then
    echo "ERROR: Main data path does not exist: ${DATA_PATH}"
    exit 1
fi

if [ ${STEER_TASK} = 'True' ] && [ ! -d "${DATA_PATH_LEARNER}" ]; then
    echo "ERROR: STEeR data path does not exist: ${DATA_PATH_LEARNER}"
    exit 1
fi

if [ ${SPEAKER_IDENTIFICATION_TASK} = 'True' ] && [ ! -d "${DATA_SPEAKER_PATH}" ]; then
    echo "ERROR: Speaker data path does not exist: ${DATA_SPEAKER_PATH}"
    exit 1
fi

if [ ${ROLE_PLAYING_TASK} = 'True' ] && [ ! -d "${DATA_ROLEPLAY_PATH}" ]; then
    echo "ERROR: Role-playing data path does not exist: ${DATA_ROLEPLAY_PATH}"
    exit 1
fi

# ==============================================================================
# TRAINING EXECUTION
# ==============================================================================

echo "============================ CoE Training Started ============================"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET}"  
echo "LoRA Fine-tuning: Enabled"
echo "Total Batch Size: ${TOTAL_BATCH_SIZE}"
echo "=============================================================================="

# ==================== ROLE-PLAYING TASK ====================
if [ ${ROLE_PLAYING_TASK} = 'True' ]; then
    echo "Starting Role-Playing Task Training..."
    
    ROLEPLAY_OUTPUT_DIR=./experiments/${MODEL_NAME}/lora/${DATASET}/RolePlay/LR_${LEARNING_RATE}_BS_${TOTAL_BATCH_SIZE}_EP_${ROLE_PLAYING_EPOCHS}
    mkdir -p "${ROLEPLAY_OUTPUT_DIR}"
    
    deepspeed --master_port=${MASTER_PORT} ../src/erc_trainer.py \
        --dataset "${DATASET}" \
        --model_name_or_path ${MODEL_PATH} \
        --data_dir ${DATA_ROLEPLAY_PATH} \
        --output_dir ${ROLEPLAY_OUTPUT_DIR} \
        --max_length ${MAX_LENGTH} \
        --batch_size ${TOTAL_BATCH_SIZE} \
        --deepspeed_config ../config/deepspeed_config.json \
        --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
        --eval_batch_size 8 \
        --num_train_epochs ${ROLE_PLAYING_EPOCHS} \
        --save_steps 100000 \
        --lora True \
        --learning_rate ${LEARNING_RATE} \
        --do_train True \
        --do_eval True \
        --statistic_mode False \
        --gradient_checkpointing

    echo "Role-Playing Task Completed!"
fi

# ==================== SPEAKER IDENTIFICATION TASK ====================
if [ ${SPEAKER_IDENTIFICATION_TASK} = 'True' ]; then
    echo "Starting Speaker Identification Task Training..."
    
    SPEAKER_OUTPUT_DIR=./experiments/${MODEL_NAME}/lora/${DATASET}/Speaker/LR_${LEARNING_RATE}_BS_${TOTAL_BATCH_SIZE}_EP_${SPEAKER_EPOCHS}
    mkdir -p "${SPEAKER_OUTPUT_DIR}"
    
    # Set checkpoint dir if role-playing was completed
    CHECKPOINT_ARG=""
    if [ ${ROLE_PLAYING_TASK} = 'True' ]; then
        CHECKPOINT_ARG="--checkpoint_dir ${ROLEPLAY_OUTPUT_DIR}"
    fi
    
    deepspeed --master_port=${MASTER_PORT} ../src/erc_trainer.py \
        --dataset "${DATASET}" \
        --model_name_or_path ${MODEL_PATH} \
        --data_dir ${DATA_SPEAKER_PATH} \
        --output_dir ${SPEAKER_OUTPUT_DIR} \
        --max_length ${MAX_LENGTH} \
        --batch_size ${TOTAL_BATCH_SIZE} \
        --deepspeed_config ../config/deepspeed_config.json \
        --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
        --eval_batch_size 8 \
        --num_train_epochs ${SPEAKER_EPOCHS} \
        --save_steps 100000 \
        --lora True \
        --learning_rate ${LEARNING_RATE} \
        --do_train True \
        --do_eval True \
        --statistic_mode False \
        ${CHECKPOINT_ARG} \
        --gradient_checkpointing

    echo "Speaker Identification Task Completed!"
fi

# ==================== STEeR (Self-Taught Emotion Reasoning) TASK ====================
if [ ${STEER_TASK} = 'True' ]; then
    echo "Starting STEeR (Self-Taught Emotion Reasoning) Task Training..."
    
    STEER_OUTPUT_DIR=./experiments/${MODEL_NAME}/lora/${DATASET}/STEeR/LR_${LEARNING_RATE}_BS_${TOTAL_BATCH_SIZE}_EP_${STEER_EPOCHS}
    mkdir -p "${STEER_OUTPUT_DIR}"
    
    # Initialize memory and training files
    trainbegin_file="${DATA_PATH_LEARNER}/trainbegin.json"
    memory_file="${STEER_OUTPUT_DIR}/memory.json"
    train_file="${DATA_PATH_LEARNER}/train.json"
    train4infer_file="${STEER_OUTPUT_DIR}/train4infer.json"
    
    echo "Initializing STEeR training files..."
    cp "${trainbegin_file}" "${memory_file}"
    cp "${train_file}" "${train4infer_file}"
    
    # Set checkpoint dir if previous tasks were completed
    CHECKPOINT_ARG=""
    if [ ${SPEAKER_IDENTIFICATION_TASK} = 'True' ]; then
        CHECKPOINT_ARG="--checkpoint_dir ${SPEAKER_OUTPUT_DIR}"
    elif [ ${ROLE_PLAYING_TASK} = 'True' ]; then
        CHECKPOINT_ARG="--checkpoint_dir ${ROLEPLAY_OUTPUT_DIR}"
    fi
    
    deepspeed --master_port=${MASTER_PORT} ../src/steer_trainer.py \
        --dataset "${DATASET}" \
        --model_name_or_path ${MODEL_PATH} \
        --data_dir ${DATA_PATH_LEARNER} \
        --output_dir ${STEER_OUTPUT_DIR} \
        --max_length ${MAX_LENGTH} \
        --batch_size ${TOTAL_BATCH_SIZE} \
        --deepspeed_config ../config/deepspeed_config.json \
        --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
        --eval_batch_size 8 \
        --num_train_epochs ${STEER_EPOCHS} \
        --save_steps 100000 \
        --lora True \
        --learning_rate ${LEARNING_RATE} \
        --do_train True \
        --do_eval True \
        --statistic_mode False \
        ${CHECKPOINT_ARG} \
        --gradient_checkpointing

    echo "STEeR Task Completed!"
    FINAL_CHECKPOINT_DIR=${STEER_OUTPUT_DIR}
fi

# ==================== FINAL ERC TASK ====================
echo "Starting Final ERC Task Training..."

FINAL_OUTPUT_DIR=./experiments/${MODEL_NAME}/lora/${DATASET}/CoE_Final/LR_${LEARNING_RATE}_BS_${TOTAL_BATCH_SIZE}_EP_${ERC_EPOCHS}
mkdir -p "${FINAL_OUTPUT_DIR}"

# Use the last completed task as checkpoint
FINAL_CHECKPOINT_ARG=""
if [ ${STEER_TASK} = 'True' ]; then
    FINAL_CHECKPOINT_ARG="--checkpoint_dir ${STEER_OUTPUT_DIR}"
elif [ ${SPEAKER_IDENTIFICATION_TASK} = 'True' ]; then
    FINAL_CHECKPOINT_ARG="--checkpoint_dir ${SPEAKER_OUTPUT_DIR}"
elif [ ${ROLE_PLAYING_TASK} = 'True' ]; then
    FINAL_CHECKPOINT_ARG="--checkpoint_dir ${ROLEPLAY_OUTPUT_DIR}"
fi

deepspeed --master_port=${MASTER_PORT} ../src/erc_trainer.py \
    --dataset "${DATASET}" \
    --model_name_or_path ${MODEL_PATH} \
    --data_dir ${DATA_PATH} \
    --output_dir ${FINAL_OUTPUT_DIR} \
    --max_length ${MAX_LENGTH} \
    --batch_size ${TOTAL_BATCH_SIZE} \
    --deepspeed_config ../config/deepspeed_config.json \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_batch_size 8 \
    --num_train_epochs ${ERC_EPOCHS} \
    --save_steps 100000 \
    --lora True \
    --learning_rate ${LEARNING_RATE} \
    --do_train True \
    --do_eval True \
    --statistic_mode True \
    ${FINAL_CHECKPOINT_ARG} \
    --gradient_checkpointing

echo "CoE Training Pipeline Completed Successfully!"
echo "Final results saved to: ${FINAL_OUTPUT_DIR}"
echo "=============================================================================="

# ==================== TRAINING SUMMARY ====================
echo ""
echo "TRAINING SUMMARY:"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET}"
echo "Role-Playing Task: ${ROLE_PLAYING_TASK}"
echo "Speaker Task: ${SPEAKER_IDENTIFICATION_TASK}"
echo "STEeR Task: ${STEER_TASK}"
echo "Final Output: ${FINAL_OUTPUT_DIR}"
echo ""
echo "Check the results in the experiments/ directory"
echo "Evaluation metrics can be found in the *_preds_for_eval_*.text files" 