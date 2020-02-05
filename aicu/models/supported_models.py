# -*- encoding: utf-8 -*-
from transformers import BertConfig, BertTokenizer

from aicu.models.modeling_bert import BertForSequenceClassificationModel, BertForTokenClassificationModel

MODEL_CLASSES = {
    "sequence_classification": {
        "bert": (BertConfig, BertForSequenceClassificationModel, BertTokenizer),
    },
    "token_classification": {
        "bert": (BertConfig, BertForTokenClassificationModel, BertTokenizer)
    }
}
