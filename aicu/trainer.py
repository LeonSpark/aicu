# -*- encoding: utf-8 -*-
from argparse import ArgumentParser
import aicu.opts as opts
from aicu.tasks.classification.classification_model import IntentClsModel
from aicu.tasks.labeling.tagger_model import NamedEntityRecognitionModel


def _get_parser():
    parser = ArgumentParser(description='trainer.py')
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    model = IntentClsModel(args, num_labels=40)
    #model = NamedEntityRecognitionModel(args)
    model.train()
    # model.predict(['去荷兰需要换荷兰盾么'])


if __name__ == '__main__':
    main()
