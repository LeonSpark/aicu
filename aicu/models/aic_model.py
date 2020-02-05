# -*- encoding: utf-8 -*-
import logging
import os

import torch
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG,
                    handlers=[logging.StreamHandler(), logging.FileHandler('logs.txt')])


class AICModel(object):
    def get_inputs_dict(self, batch):
        inputs = {
            "input_ids":      batch[0],
            "attention_mask": batch[1],
            "labels":         batch[3]
        }
        # XLM, DistilBERT and RoBERTa don't use segment_ids
        if self.args.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.args.model_type in ["bert", "xlnet"] else None
        return inputs

    def save_model(self, model, save_dir):
        output_dir = os.path.join(self.args.output_dir, save_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        logger.info('Saving model checkpoint to %s', output_dir)
