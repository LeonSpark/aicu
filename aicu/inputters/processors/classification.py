# -*- encoding: utf-8 -*-
import logging
import os
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from .base_processor import DataProcessor

from ..base_data import InputExample, InputFeatures

logger = logging.getLogger(__name__)


class IntentClsProcessor(DataProcessor):
    """Processor for the sequence classification data set.
    Example file format:
        这是类型1  a
        这是类型2  b
        这是类型3  c
        a,b,c can also be number
    """
    def __init__(self, num_labels):
        self.num_labels = num_labels

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return list(map(str, range(self.num_labels)))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[0]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def convert_example_to_feature(example_row):
    example, max_seq_len, tokenizer, task, label_list, pad_on_left, pad_token, pad_token_segment_id, mask_padding_with_zero = example_row
    if task is not None:
        processor = classifier_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
    label_map = {label: i for i, label in enumerate(label_list)}
    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        add_special_tokens=True,
        max_length=max_seq_len
    )
    input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_len - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
    assert len(attention_mask) == max_seq_len, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                         max_seq_len)
    assert len(token_type_ids) == max_seq_len, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                         max_seq_len)
    label = label_map.get(example.label, 0)

    return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label)


def convert_examples_to_features(
        examples,
        max_seq_len,
        tokenizer,
        task,
        label_list,
        pad_on_left,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        process_count=cpu_count() - 1,
        silent=False,
        chunk_size=500,
        use_multiprocessing=True
):
    if not label_list:
        label_list = list(set(ex.label for ex in examples))
    examples = [(ex, max_seq_len, tokenizer, task, label_list, pad_on_left, pad_token, pad_token_segment_id,
                 mask_padding_with_zero) for ex in examples]
    if use_multiprocessing:
        with Pool(process_count) as p:
            features = list(tqdm(p.imap(convert_example_to_feature, examples, chunksize=chunk_size),
                                 total=len(examples), disable=silent))
    else:
        features = [convert_example_to_feature(ex) for ex in tqdm(examples, disable=silent)]
    return features


classifier_processors = {
    "sequence_classification": IntentClsProcessor
}
