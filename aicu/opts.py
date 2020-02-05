# -*- encoding: utf-8 -*-

from transformers import BertConfig


def model_opts(parser):
    group = parser.add_argument_group('Model')

    group.add_argument("--data_dir", "-data_dir", type=str, required=True,
              help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    group.add_argument("--model_type", "-model_type", default=None, type=str, required=True,
              help="Model type selected in the list: bert")
    group.add_argument("--model_name_or_path", default=None, type=str, required=True,
              help="Path to pre-trained model or shortcut name selected in the list: " +
                   ",".join(BertConfig.pretrained_config_archive_map.keys()))
    group.add_argument("--task_name", default=None, type=str, required=True,
              help="The name of the task to train selected in the list: bert")
    group.add_argument("--output_dir", default=None, type=str, required=True,
              help="The output directory where the model predictions and checkpoints will be written.")
    group.add_argument("--lstm_layers", default=0, type=int,
              help="Number of layers of lstm, 0 means no lstm layer.")
    group.add_argument("--lstm_cell_type", type=str, default='gru',
              help="Lstm cell type, one of gru | lstm")
    group.add_argument("--lstm_hidden_size", default=1024, type=int,
              help="Lstm hidden units")
    group.add_argument("--lstm_dropout", default=0.2, type=float,
              help="Lstm cell dropout")


def train_opts(parser):
    """Training and saving options"""
    group = parser.add_argument_group("Train")
    group.add_argument("--cache_dir", default=".cache", type=str,
              help="Where do you want to store the pre-trained models downloaded from s3")
    group.add_argument("--max_seq_len", default=64, type=int,
              help="The maximum total input sequence length after tokenization. Sequences longer "
                   "than this will be truncated, sequences shorter will be padded.")
    group.add_argument("--do_train", action='store_true',
              help="Whether to run training")
    group.add_argument("--do_eval", action="store_true",
              help="Whether to run eval on the dev set.")
    group.add_argument("--evaluate_during_training", action='store_true',
              help="Rul evaluation during training at each logging step.")
    group.add_argument("--batch_size", default=8, type=int,
              help="Batch size per GPU/CPU for training.")
    group.add_argument("--eval_batch_size", default=8, type=int,
              help="Batch size per GPU/CPU for evaluation.")
    group.add_argument('--gradient_accumulation_steps', type=int, default=1,
              help="Number of updates steps to accumulate before performing a backward/update pass.")
    group.add_argument("--learning_rate", default=5e-5, type=float,
              help="The initial learning rate for Adam.")
    group.add_argument("--weight_decay", default=0.0, type=float,
              help="Weight decay if we apply some.")
    group.add_argument("--adam_epsilon", default=1e-8, type=float,
              help="Epsilon for Adam optimizer.")
    group.add_argument("--max_grad_norm", default=1.0, type=float,
              help="Max gradient norm.")
    group.add_argument("--num_train_epochs", default=3.0, type=float,
              help="Total number of training epochs to perform.")
    group.add_argument("--max_steps", default=-1, type=int,
              help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    group.add_argument("--warmup_steps", default=0, type=int,
              help="Linear warmup over warmup_steps.")

    group.add_argument("--logging_steps", type=int, default=50,
              help="Log every X updates steps.")
    group.add_argument("--save_steps", type=int, default=50,
              help="Save checkpoint every X updates steps.")
    group.add_argument("--no_cuda", action='store_true',
              help="Avoid using CUDA when available")
    group.add_argument('--overwrite_output_dir', action='store_true',
              help="Overwrite the content of the output directory")
    group.add_argument('--overwrite_cache', action='store_true',
              help="Overwrite the cached training and evaluation sets")
    group.add_argument('--seed', type=int, default=42,
              help="random seed for initialization")

    group.add_argument('--fp16', action='store_true',
              help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    group.add_argument('--fp16_opt_level', type=str, default='O1',
              help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                   "See details at https://nvidia.github.io/apex/amp.html")
    group.add_argument("--local_rank", type=int, default=-1,
              help="For distributed training: local_rank")
    group.add_argument("--n_gpu", type=int, default=1,
              help="number of gpu")
    group.add_argument("--ext_data", type=str, default="data/intent/poi.txt",
              help="poi names")
