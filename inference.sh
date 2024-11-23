MODEL_PATH = "$HOME/bin/PaliGemma/assets/paligemma-3b-pt-224"
PROMPT = "Where is the person resting?"
IMAGE_FILE_PATH = "$HOME/bin/PaliGemma/assets/test.png"
MAX_TOKENS = 100
TEMPERATURE = 0.8
TOP_P = 0.9
DO_SAMPLE = False
ONLY_CPU = False

python $HOME/bin/PaliGemma/gemma/inference.py \
    --model_path $MODEL_PATH \
    --prompt "$PROMPT" \
    --image_file_path $IMAGE_FILE_PATH \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU