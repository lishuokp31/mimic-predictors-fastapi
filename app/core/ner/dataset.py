from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from .utils import Entity, parse_entities, align_labels
from typing import Tuple, List


class MedicalNERDataset(Dataset):
    """
    数据集合类，视数据的情况看是否需要划分，将处理后的训练集和验证集返回
    """
    def __init__(self, dataset_dir: str, uncased=True, transform=None):
        self.dataset_path = Path(dataset_dir)
        self.uncased = uncased
        self.files = list(self.dataset_path.rglob('*.txt'))
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[str, List[Entity]]:
        f = self.files[index]
        label_path = self.dataset_path / (f.name[:-4] + '.ann')
        assert label_path.is_file()

        # load text sample
        with f.open('r', encoding='utf-8') as fd:
            sequence = fd.read().lower() if self.uncased else fd.read()

        # load text sample entity labels and parse it
        # 加载文本样本实体标签并对其进行解析
        with label_path.open('r', encoding='utf-8') as fd:
            # entities 是读取 ann 文件并排序后的 label
            entities = sorted(parse_entities(fd), key=lambda x: x.start)
            outputs = sequence, entities

        # apply transforms if present
        # 应用变换
        if self.transform:
            outputs = self.transform(sequence, entities)

        return f, outputs

    def __len__(self) -> int:
        return len(self.files)


class AlignLabels:
    """
    标签对齐
    """
    def __init__(self, tokenizer: BertTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(
        self,
        sequence: str,
        entities: List[Entity],
    ) -> List[Tuple[str, List[Entity]]]:
        # tokenize the input sequence (without special tokens, e.g. [CLS], [SEP])
        # 标记输入序列（不带特殊标记，例如[CLS]，[SEP]）
        tokens = self.tokenizer.tokenize(sequence, add_special_tokens=False)

        # reposition entity labels' start and end indices
        # according to the new indices of the tokens list.
        # Also, the chosen BERT variant is uncased (capitalization is ignored)
        # 根据令牌列表的新索引重新定位实体标签的开始索引和结束索引。 此外，所选的BERT变体是无大小写的（忽略大小写）
        labels = align_labels(
            sequence=sequence,
            tokens=tokens,
            entities=entities,
            vocab=self.tokenizer.get_vocab(),
        )

        return sequence, labels
