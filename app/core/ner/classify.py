from .constants import BERT_VARIANT
from .modeling import classify
from pathlib import Path
from transformers import BertForTokenClassification, BertTokenizerFast

import fire


def do_classify(sequence: str):
    """
    进行实体识别，顶层
    """
    # initialize checkpoint path
    # ckpt_path = Path(f'finetuned-{BERT_VARIANT}'.replace('/', '-'))
    # ckpt_path = Path('../services/finetuned-bert-medical-ner/finetuned-bert-medical-ner')
    ckpt_path = Path('core/ner/finetuned-bert-medical-ner')
    # ckpt_path = Path('core/ner/finetuned-bert-medical-ner')

    
    assert ckpt_path.is_dir(), 'Finetune model first using finetune.py'

    # load finetuned model
    # 加载训练好的模型
    tokenizer = BertTokenizerFast.from_pretrained(BERT_VARIANT)
    model = BertForTokenClassification.from_pretrained(ckpt_path).eval()

    # print output
    # 开始实体识别
    entities = classify(model, tokenizer, sequence)
    for entity in entities:
        print(entity)
    return entities


if __name__ == '__main__':
    fire.Fire(do_classify)
