from .dataset import MedicalNERDataset
from transformers import BertForTokenClassification, BertTokenizerFast
from typing import Dict, List
from .utils import (
    Entity,
    split_into_blocks,
    extract_entities,
    realign_extracted_entities,
)

import numpy as np
import torch


def classify(
    model: BertForTokenClassification,
    tokenizer: BertTokenizerFast,
    # 输入
    sequence: str,
    labels: List[int] = None,
) -> List[Entity]:
    """
    classify的功能：
    给定model、tokenizer，sequence：
    给定label(对应以下label不为空的情况)，计算loss；
    label为空：预测entities

    训练好的模型，可以直接使用
    """
    # ensure model is configured to return dict
    # otherwise this code will break
    # 确保模型配置为返回dict，否则此代码将中断
    if not model.config.return_dict:
        raise ValueError(
            'Model should be instantiated with `return_dict=True`')

    # convert input sequence (and optional labels) into an inputs bundle
    # 将输入序列（和可选标签）转换为输入包
    inputs, mask = pack_sequence_as_inputs(
        tokenizer=tokenizer,
        sequence=sequence,
        labels=labels,
        max_token_length=model.config.max_position_embeddings,
    )

    # put data on the gpu (if available)
    # if torch.cuda.is_available():
    #     model.cuda()
    #     inputs = {k: v.cuda() for k, v in inputs.items()}

    # if labels is not None, it means that the caller is interested in the loss
    # value of the given input sequence. So, this should be done in a grad context.
    # 如果labels不是None，则表示调用者对给定输入序列的损失值感兴趣。所以，这应该在毕业的背景下进行。
    if labels is not None:
        return model(**inputs).loss

    # if labels is None, it means that the caller is interested in the entities
    # to be recognized by the model. In this case, the outputs can be computed
    # without a grad context
    # 如果labels为None，则表示调用者对模型要识别的实体感兴趣。在这种情况下，可以在没有梯度上下文的情况下计算输出
    with torch.no_grad():
        logits = model(**inputs).logits.cpu()

    # decode model's output
    # 解码模型输出
    entities = extract_entities(
        sequence=sequence,
        logits=logits[:, 1:-1][mask],
        encode=tokenizer.encode,
        decode=tokenizer.decode,
    )
    entities = realign_extracted_entities(
        sequence=sequence,
        tokens=tokenizer.tokenize(sequence),
        entities=entities,
        vocab=tokenizer.get_vocab(),
    )

    return list(entities)


def pack_sequence_as_inputs(
    tokenizer: BertTokenizerFast,
    sequence: str,
    labels: List[int] = None,
    max_token_length=512,
) -> Dict[str, torch.Tensor]:
    """
    将输入序列（和可选标签）转换为输入包(数据预处理)
    """
    # the chosen model is uncased (doesn't really matter since the
    # subject of the experiment are chinese texts). so, every input
    # sequence shall be transformed to lowercase
    # 选择的模型是不带大小写的（因为实验的主题是中文文本，所以没什么关系）。 因此，每个输入序列都应转换为小写
    sequence = sequence.lower()

    # using the given tokenizer, transform sequence into input IDs,
    # attention mask, and token type IDs (as tensors)
    # 使用给定的令牌生成器，将序列转换为输入ID，注意掩码和令牌类型ID（作为张量）
    inputs = tokenizer(sequence, return_tensors='pt')

    # mask is used to decode the model output later in the process
    # 掩码用于在此过程的后面对模型输出进行解码
    mask = inputs['attention_mask'][:, 1:-1].bool()
    n_tokens = inputs['input_ids'].shape[1]

    # labels for computing the token classification loss
    # if labels are defined, its length should be equal to the
    # number of tokens (excluding [CLS] and [SEP])
    # 用于定义令牌分类损失的标签（如果已定义标签），其长度应等于令牌的数量（不包括[CLS]和[SEP]）
    if labels is not None:
        inputs['labels'] = torch.tensor([0] + labels + [0])
        assert len(labels) == n_tokens - 2

    # split the input sequence into blocks
    # if its token length is greater than `max_position_embeddings`
    # 判断样本长度是否超过 bert 模型能够处理的长度
    if n_tokens > max_token_length:
        return batchify_long_inputs(
            tokenizer=tokenizer,
            inputs=inputs,
            labels=labels,
            max_token_length=max_token_length,
        )

    return inputs, mask


def batchify_long_inputs(
    tokenizer: BertTokenizerFast,
    inputs: Dict[str, torch.Tensor],
    labels: List[int] = None,
    max_token_length=512,
):
    """
    In cases where the input sequence token length is greater than 512,
    the input sequence is packed into a batch mainly for efficiency purposes.
    在输入序列token长度大于512的情况下，将输入序列打包到批次中主要是出于提高效率的目的。
    """
    token_ids = inputs['input_ids'][0, 1:-1].tolist()

    # get the special token IDs for later use
    # 获取特殊tokenID，以供以后使用
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id

    # the token IDs of the spliced sequence is collected in an array
    # in preparation for the creation of the needed matrices ahead
    # 剪接序列的tokenID被收集在一个数组中，以准备在前面创建所需的矩阵
    token_blocks, label_blocks = [], []
    for start, end in split_into_blocks(
        token_ids=token_ids,
        separator_token_id=tokenizer.get_vocab().get('。'),
        block_size=max_token_length - 2,
    ):
        # split the input tokens into blocks (separated by period)
        # 将输入tokens序列拆分为块（以句点分隔）
        token_blocks.append([cls] + token_ids[start:end] + [sep])

        # also split the labels into blocks (if passed)
        # 还将标签分成块（如果通过）
        if labels is not None:
            label_blocks.append([0] + labels[start:end] + [0])

    # create a matrix vertically stacking the token IDs.
    # the width of this matrix depends on the longest token block.
    # also, each row of the matrix contains [CLS] and [SEP] tokens.
    # 创建一个垂直堆叠token ID的矩阵。 该矩阵的宽度取决于最长的token块。 同样，矩阵的每一行都包含[CLS]和[SEP]token。
    max_block_len = max([len(block) for block in token_blocks])
    input_ids = torch.tensor([
        block + [pad] * (max_block_len - len(block))
        for block in token_blocks
    ])
    attention_mask = torch.tensor([
        [1] * len(block) + [0] * (max_block_len - len(block))
        for block in token_blocks
    ])
    label_ids = torch.tensor([
        block + [pad] * (max_block_len - len(block))
        for block in label_blocks
    ]) if labels is not None else None

    # basically the same with `attention_mask` except that it doesn't
    # take into account the [CLS] and [SEP] positions.
    # this is created so that the final logits can be indexed conveniently
    # 与“ attention_mask”基本相同，除了它不考虑[CLS]和[SEP]位置。 创建它是为了方便最终索引登录
    mask = torch.tensor([
        [1] * (len(block) - 2) + [0] * (max_block_len - len(block))
        for block in token_blocks
    ], dtype=torch.bool)

    # combine inputs as one batch to be processed at once
    # 将输入合并为一批，一次处理
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': torch.zeros_like(input_ids),
    }

    # add the labels tensor (if applicable)
    # 添加标签张量（如果适用）
    if labels is not None:
        inputs['labels'] = label_ids

    return inputs, mask


def get_metrics(model: BertForTokenClassification, tokenizer: BertTokenizerFast, dataset_dir: str):
    """
    在验证集上评价每个epoch训练得到的model
    """
    metrics = []
    # 处理数据集
    dataset = MedicalNERDataset(dataset_dir=dataset_dir, uncased=True)

    for _, (sequence, entities) in dataset:
        # get model predictions
        # 获得模型预测结果
        preds = classify(model, tokenizer, sequence)

        # 求得标签给定的entities和预测的结果的交集
        correct = set(entities).intersection(set(preds))

        # compute metrics
        # 计算评价
        precision = safe_division(len(correct), len(preds))
        recall = safe_division(len(correct), len(entities))
        f1score = safe_division(2 * precision * recall, precision + recall)
        metrics.append((precision, recall, f1score))

    # 返回验证集中所有样本的 precision、recall 和 f1score 的平均值 (忽略掉 nan)
    return np.nanmean(np.array(metrics), axis=0).tolist()


def safe_division(n, d):
    """
    除法
    """
    if n == np.nan or d == np.nan or d == 0:
        return np.nan

    return n / d
