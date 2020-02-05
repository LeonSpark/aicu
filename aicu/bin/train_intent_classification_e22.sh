#!/bin/bash
python aicu/trainer.py --data_dir data/intent --model_type bert --model_name_or_path  hfl/chinese-bert-wwm-ext --task_name sequence_classification --output_dir outputs --logging_steps  512  --num_train_epochs 100  --batch_size 16 --warmup_steps 1024 --learning_rate 5e-6   --evaluate_during_training --save_steps 1024 --do_train --do_eval

python aicu/trainer.py --data_dir data/intent --model_type bert --model_name_or_path  hfl/chinese-bert-wwm-ext --task_name sequence_classification --output_dir outputs --logging_steps  1024  --num_train_epochs 100  --batch_size 8 --warmup_steps 2048 --learning_rate 5e-6   --evaluate_during_training --save_steps 2000 --do_train --do_eval
