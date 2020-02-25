# -*- encoding: utf-8 -*-
import json
import logging
import os

import numpy as np
import torch
import jieba
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from aicu.inputters.base_data import InputExample
from aicu.inputters.processors.classification import classifier_processors, convert_examples_to_features
from aicu.models.aic_model import AICModel
from aicu.models.supported_models import MODEL_CLASSES
from aicu.utils.utils import set_seed, compute_metrics

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG,
                    handlers=[logging.StreamHandler(), logging.FileHandler('logs.txt')])


class IntentClsModel(AICModel):
    def __init__(self, args, num_labels=2, use_cuda=True):
        self.type = 'sequence_classification'
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.type][self.args.model_type]
        self.num_labels = num_labels 
        task_name = self.args.task_name.lower()
        self.task = task_name
        self.processor = classifier_processors[task_name](num_labels)
        self.label_list = self.processor.get_labels()
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise ValueError("'use_cuda' set to True when cuda is unavailable.")
        else:
            self.device = 'cpu'
        self.tokenizer = tokenizer_class.from_pretrained(self.args.model_name_or_path,
                                                         cache_dir=self.args.cache_dir)
        self.config = config_class.from_pretrained(self.args.model_name_or_path, num_labels=self.num_labels,
                                                   finetune_task=task_name,
                                                   cache_dir=self.args.cache_dir)
        self.model = model_class.from_pretrained(self.args.model_name_or_path,
                                                 config=self.config,
                                                 cache_dir=self.args.cache_dir)
        self.model.to(self.device)

        logger.info("Training/evaluation params %s", self.args)

    def load_and_cache_examples(self, examples, mode, do_cache=True, multi_processing=True):
        if not os.path.isdir(self.args.cache_dir):
            os.mkdir(self.args.cache_dir)
        cache_feature_file = os.path.join(self.args.cache_dir,
                                          'cached_{}_{}_{}_{}_{}'.format(
                                              mode, self.args.model_type, self.args.max_seq_len, self.num_labels,
                                              len(examples)
                                          ))
        if os.path.exists(cache_feature_file) and not self.args.overwrite_cache and do_cache:
            features = torch.load(cache_feature_file)
            logger.info("Load features from cache file {}.".format(cache_feature_file))
        else:
            logger.info('Converting to features started.')
            features = convert_examples_to_features(
                examples,
                max_seq_len=self.args.max_seq_len,
                tokenizer=self.tokenizer,
                task=self.task,
                label_list=self.label_list,
                pad_on_left=bool(self.args.model_type in ['xlnet']),
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0,
                use_multiprocessing=multi_processing
               )
            if do_cache:
                torch.save(features, cache_feature_file)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    def train(self):
        args = self.args
        model = self.model
        train_examples = self.processor.get_train_examples(args.data_dir)
        train_dataset = self.load_and_cache_examples(train_examples, 'train')
        tf_board_writer = SummaryWriter()
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
        if args.max_steps > 0:
            steps_total = args.max_steps
            args.num_train_epochs = steps_total // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            steps_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=steps_total)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)

        global_step = 0
        tr_loss, logging_loss, best_acc = 0.0, 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])
        set_seed(args)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration', disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = self.get_inputs_dict(batch)
                outputs = model(**inputs)
                loss = outputs[0]
                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logs = {}
                        if args.local_rank == -1 and args.eval_batch_size:
                            results, model_outputs, ground_truth = self.evaluate()
                            if results['acc'] > best_acc:
                                self.save_model(model, 'checkpoint-best')
                                best_acc = results['acc']
                            for key, value in results.items():
                                eval_key = 'eval_{}'.format(key)
                                logs[eval_key] = value
                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                       # learning_rate_scalar = args.learning_rate
                        logs['learning_rate'] = learning_rate_scalar 
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            tf_board_writer.add_scalar(key, value, global_step)
                        logger.info(json.dumps({**logs, **{'step': global_step}}) + "\n")
                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        self.save_model(model, 'checkpoint-{}'.format(global_step))

                if 0 < args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < args.max_steps < global_step:
                train_iterator.close()
                break
        self.save_model(model, 'model_final')
        if args.local_rank in [-1, 0]:
            tf_board_writer.close()
        return global_step, tr_loss / global_step

    def save_model(self, model, save_dir):
        output_dir = os.path.join(self.args.output_dir, save_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        logger.info('Saving model checkpoint to %s', output_dir)

    def evaluate(self):
        model = self.model
        args = self.args
        eval_dataset = self.load_and_cache_examples(self.processor.get_dev_examples(args.data_dir), 'eval')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        preds = None
        output_label_ids = None
        model.eval()
        for batch in tqdm(eval_dataloader, disable=False):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self.get_inputs_dict(batch)
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
            if preds is None:
                preds = logits.detach().cpu().numpy()
                output_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                output_label_ids = np.append(output_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(self.task, preds, output_label_ids)
        return result, preds, output_label_ids

    # def _get_inputs_dict(self, batch):
    #     inputs = {
    #         "input_ids":      batch[0],
    #         "attention_mask": batch[1],
    #         "labels":         batch[3]
    #     }
    #     # XLM, DistilBERT and RoBERTa don't use segment_ids
    #     if self.args.model_type != "distilbert":
    #         inputs["token_type_ids"] = batch[2] if self.args.model_type in ["bert", "xlnet"] else None
    #
    #     return inputs

    def predict(self, to_predict):
        """predict on the list of text
        :param to_predict: list of text to predict
        :return: A python list of the predictions for each text
        """
        model = self.model
        args = self.args
        pois = list(map(str.strip, open(self.args.ext_data, 'r').readlines()))
        poi = dict(zip(pois, range(len(pois))))
        eval_example = []
        for i, text in enumerate(to_predict):
            seg_list = list(filter(lambda x: len(x) >= 3, jieba.cut_for_search(text)))
            if any(x in poi and len(x) >= int(len(text)/2) for x in seg_list):
                eval_example.append(InputExample(i, text, None, "19"))
            else:
                eval_example.append(InputExample(i, text, None, "0"))
        # eval_example = [InputExample(i, text, None, "0") for i, text in enumerate(to_predict)]
        eval_dataset = self.load_and_cache_examples(eval_example, 'test', do_cache=False, multi_processing=False)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        preds = None
        model.eval()
        for batch in tqdm(eval_dataloader, disable=True):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self.get_inputs_dict(batch)
                outputs = model(**inputs)
                _, logits = outputs[:2]
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        preds = [pred if example.label == "0" else int(example.label) for pred, example in zip(preds, eval_example)]
        return preds, model_outputs
