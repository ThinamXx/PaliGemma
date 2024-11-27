export MODEL_PATH="/home/ubuntu/bin/PaliGemma/assets/paligemma-3b-pt-224"
export PROMPT="Where is the person resting?"
export IMAGE_FILE_PATH="/home/ubuntu/bin/PaliGemma/assets/test.png"
export MAX_TOKENS=500
export TEMPERATURE=0.8
export TOP_P=0.9
export DO_SAMPLE=False
export ONLY_CPU=False

python /home/ubuntu/bin/PaliGemma/gemma/inference.py \
    --model_path $MODEL_PATH \
    --prompt "$PROMPT" \
    --image_file_path $IMAGE_FILE_PATH \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU