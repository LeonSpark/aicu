# -*- encoding: utf-8 -*-
import logging
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import precision_score, recall_score, f1_score
from aicu.inputters.processors.labeling import processors, convert_examples_to_features
from aicu.inputters.base_data import NERInputExample
from aicu.models.aic_model import AICModel
from aicu.models.supported_models import MODEL_CLASSES
from aicu.utils.utils import set_seed, compute_metrics

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG,
                    handlers=[logging.StreamHandler(), logging.FileHandler('logs.txt')])


class NamedEntityRecognitionModel(AICModel):
    def __init__(self, args, use_cuda=True):
        self.args = args
        self.type = 'token_classification'
        self.task = args.task_name.lower()
        self.processor = processors[self.task]()
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.pad_token_label_id = 0 
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.type][args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        self.config = config_class.from_pretrained(args.model_name_or_path,
                                                   num_labels=self.num_labels,
                                                   cache_dir=args.cache_dir)
        self.model = model_class.from_pretrained(args.model_name_or_path,
                                                 config=self.config,
                                                 args=self.args,
                                                 cache_dir=self.args.cache_dir)
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise ValueError("'use_cuda' set to True when cuda is unavailable.")
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        logger.info('Training/Evaluation params %s', self.args)

    def load_and_cache_examples(self, examples, do_cache=True, multi_processing=True):
        tokenizer = self.tokenizer
        if not os.path.isdir(self.args.cache_dir):
            os.mkdir(self.args.cache_dir)
        mode = 'train' if self.args.do_train else 'dev'
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
                self.label_list,
                max_seq_len=self.args.max_seq_len,
                tokenizer=self.tokenizer,
                cls_token_at_end=bool(self.args.model_type in ['xlnet']),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(self.args.model_type in ['roberta']),
                pad_on_left=bool(self.args.model_type in ['xlnet']),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0,
                pad_token_label_id=self.pad_token_label_id
            )
            if do_cache:
                torch.save(features, cache_feature_file)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    def train(self):
        args = self.args
        model = self.model
        train_examples = self.processor.get_train_examples(args.data_dir)
        train_dataset = self.load_and_cache_examples(train_examples)
        sw = SummaryWriter()
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
        steps_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=steps_total)
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
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=False)
        set_seed(args)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration', disable=False)
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
                    with amp.scale_loss(loss, optimizer) as scale_loss:
                        scale_loss.backward()
                else:
                    loss.backward()
                tr_loss = loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    #   since pytroch 1.1.0 https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
                    optimizer.step()
                    scheduler.step()    # update lr
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        if args.local_rank == -1 and args.evaluate_during_training:
                            results, _, _ = self.evaluate()
                            for key, value in results.items():
                                sw.add_scalar('eval_{}'.format(key), value, global_step)
                        sw.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        sw.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        self.save_model(model, 'checkpoint-{}'.format(global_step))
                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break
        if args.local_rank in [-1, 0]:
            sw.close()
        self.save_model(model, 'final-model')
        return global_step, tr_loss / global_step

    def evaluate(self):
        model = self.model
        args = self.args
        eval_examples = self.processor.get_dev_examples(args.data_dir)
        eval_dataset = self.load_and_cache_examples(eval_examples, do_cache=True)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        preds = None
        out_label_ids = None
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        for batch in tqdm(eval_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = self.get_inputs_dict(batch)
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                if args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        
        preds = np.argmax(preds, axis=2)
        label_map = {i: label for i, label in enumerate(self.label_list)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])  

        results = {
            'loss': eval_loss,
            'precision': precision_score(out_label_list, preds_list),
            'recall': recall_score(out_label_list, preds_list),
            'f1': f1_score(out_label_list, preds_list)
        }
        logger.info("***** Eval results  *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        return results, preds_list, out_label_list

    def predict(self, to_predict):
        model = self.model
        args = self.args
        tokenizer = self.tokenizer
        device = self.device
        predict_examples = [NERInputExample(guid='predict-{}'.format(i),
                                            words=tokenizer.tokenize(sent),
                                            labels=['O'] * len(tokenizer.tokenize(sent)))
                            for i, sent in enumerate(to_predict)]
        predict_dataset = self.load_and_cache_examples(predict_examples, do_cache=False, multi_processing=False)
        predict_sampler = SequentialSampler(predict_dataset)
        predict_dataloader = DataLoader(predict_dataset, sampler=predict_sampler, batch_size=args.eval_batch_size)
        preds = None
        out_label_ids = None
        model.eval()
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = self.get_inputs_dict(batch)
                outputs = model(**inputs)
                _, logits = outputs[:2]
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        model_output = preds
        preds = np.argmax(preds, axis=-1)
        label_map = {i: label for i, label in enumerate(self.label_list)}
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    preds_list[i].append(label_map[preds[i][j]])
        preds = [[(word, preds_list[i][j]) for j, word in enumerate(tokenizer.tokenize(sentence)[:len(preds_list[i])])] for i, sentence in enumerate(to_predict)]
        return preds, model_output

