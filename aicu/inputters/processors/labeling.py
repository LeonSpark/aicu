# -*- encoding: utf-8 -*-
import copy
import logging
import os

from tqdm import tqdm

from .base_processor import DataProcessor
from multiprocessing import cpu_count, Pool

from ..base_data import NERInputExample, NERInputFeatures

logger = logging.getLogger(__name__)


class NERProcessor(DataProcessor):
    """Processor for the labeling data set.
    Example:
        中   B-ORG
        国   I-ORG
        的   O
        空   O
        气   O
        真   O
        甜   O

        你   O
        说   O
        呢   O
    """

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
        return ["X", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        words = []
        labels = []
        guid_index = 1
        for (i, line) in enumerate(lines):
            if not line:
                if words:
                    examples.append(NERInputExample(guid="{}-{}".format(set_type, guid_index),
                                                    words=words,
                                                    labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                words.append(line[0])
                if len(line) > 1 and line[1].strip():
                    labels.append(line[1])
                else:
                    labels.append('O')
        if words:
            examples.append(NERInputExample(guid="{}-{}".format(set_type, guid_index),
                                            words=words,
                                            labels=labels))
        return examples


def convert_example_to_feature(example_row):
    example, label_map, max_seq_len, tokenizer, cls_token_at_end, cls_token, cls_token_segment_id, sep_token, sep_token_extra, pad_on_left, pad_token, pad_token_segment_id, pad_token_label_id, seq_a_segment_id, mask_padding_with_zero = example_row
    tokens = []
    label_ids = []
    for word, label in zip(example.words, example.labels):
        words_tokens = tokenizer.tokenize(word)
        tokens.extend(words_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        try:

            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(words_tokens) - 1))
        except Exception as e:
            print('words: {} - label:{}'.format(word, label))        
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_len - special_tokens_count:
        tokens = tokens[:(max_seq_len - special_tokens_count)]
        label_ids = label_ids[:(max_seq_len - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
    segment_ids = [seq_a_segment_id] * len(tokens)
    if cls_token_at_end:
        tokens += [cls_token]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    padding_length = max_seq_len - len(input_ids)
    if pad_on_left:
        input_ids = ([[pad_token] * padding_length]) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
    else:
        input_ids += ([pad_token] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([pad_token_segment_id] * padding_length)
        label_ids += ([pad_token_label_id] * padding_length)
    return NERInputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_ids=label_ids)


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_len,
        tokenizer,
        cls_token_at_end=False,
        cls_token='[CLS]',
        cls_token_segment_id=1,
        sep_token='[SEP]',
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        seq_a_segment_id=0,
        mask_padding_with_zero=True,
        process_count=cpu_count() - 2,
        chunk_size=500,
        silent=False
):
    label_map = {label: i for i, label in enumerate(label_list)}
    examples = [(
        example,
        label_map,
        max_seq_len,
        tokenizer,
        cls_token_at_end,
        cls_token,
        cls_token_segment_id,
        sep_token,
        sep_token_extra,
        pad_on_left,
        pad_token,
        pad_token_segment_id,
        pad_token_label_id,
        seq_a_segment_id,
        mask_padding_with_zero
    ) for example in examples]
    with Pool(process_count) as p:
        features = list(tqdm(p.imap(convert_example_to_feature, examples, chunksize=chunk_size), total=len(examples), disable=silent))
    return features


processors = {
    "labeling": NERProcessor
}
