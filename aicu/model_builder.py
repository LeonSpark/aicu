# -*- encoding: utf-8 -*-
from transformers import BertConfig, BertTokenizer

from .models.modeling_bert import BertForSequenceClassificationModel


def build_model(model_opts, checkpoint):
    pass