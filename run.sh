export lr=3e-5
export s=100
echo "${lr}"
export MODEL_DIR=phobert-crf
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c"/"$s
echo "${MODEL_DIR}"
python3 main.py --token_level word \
                  --model_type phobert \
                  --model_dir $MODEL_DIR \
                  --data_dir PhoNER_COVID19/data \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 50 \
                  --tuning_metric slot_f1 \
                  --use_crf \
                  --gpu_id 0 \
                  --learning_rate $lr
