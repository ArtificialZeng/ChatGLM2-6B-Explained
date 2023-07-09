# CSDN彩色版：
#ChatGLM2-6B源码解析./ptuning/main.py （一）  https://zengxiaojian.blog.csdn.net/article/details/131617133?spm=1001.2014.3001.5502

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset  #从 Hugging Face 的 datasets 库中导入 load_dataset 函数，用于加载各种预处理后的数据集。
import jieba 
from rouge_chinese import Rouge  #从 rouge_chinese 模块中导入 Rouge 类，这个类可以用来计算 Rouge 分数，它是一种用来评估机器生成文本（如机器翻译或文本摘要）与人类参考文本之间相似度的指标。
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  #从 nltk.translate.bleu_score 模块中导入 sentence_bleu 和 SmoothingFunction。sentence_bleu 是用来计算单个句子的 BLEU 分数的函数，
#而 SmoothingFunction 是用来处理BLEU分数计算过程中出现的0分情况。
import torch

#导入了 transformers 库及其一些子模块。transformers 库提供了许多预训练的神经网络模型，可以用于各种自然语言处理任务。
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
#从 trainer_seq2seq 模块导入 Seq2SeqTrainer 类，这个类是用来训练序列到序列（seq2seq）模型的。
from trainer_seq2seq import Seq2SeqTrainer

from arguments import ModelArguments, DataTrainingArguments #这行代码从 arguments 模块导入了两个类，这两个类用于解析和处理命令行参数。

logger = logging.getLogger(__name__)  #创建一个记录器（logger），这个记录器可以用来记录脚本的运行情况。

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))  #创建一个 HfArgumentParser 对象，它将解析和处理 ModelArguments、DataTrainingArguments 和 Seq2SeqTrainingArguments 这三个类的实例。
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):  #检查脚本的命令行参数是否为一个 .json 文件。如果是，那么将会从这个文件中读取参数。
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        #读取 .json 文件中的参数，并将其分别赋值给 model_args、data_args 和 training_args。
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        #如果命令行参数不是一个 .json 文件，那么这行代码将会直接从命令行参数中解析出参数。
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    #设置日志的基础配置，包括日志的格式、日期格式以及处理器。
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    #如果 training_args.should_log 为真（即需要记录日志），那么设置日志等级为 info。
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    #设置 logger 的日志等级。
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # Finetune
        model = model.float()

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train: 
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column  #从数据参数中获取提示列名称，也就是用于提问的列。
    response_column = data_args.response_column  #从数据参数中获取回答列的名称，也就是作为回答或目标的列。
    history_column = data_args.history_column  #从数据参数中获取历史对话列的名称，如果存在的话，这些历史对话将被用作提问的上下文。
    
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    #以下是预处理函数，它们用于将输入和目标列进行格式化和分词。格式化的结果将被用于模型的训练和验证。
    def preprocess_function_eval(examples):  #和preprocess_function_train这两个函数是为评估和训练准备数据的。它们从示例数据中提取问题和回答，并根据需要将其进行格式化和分词。然后它们会将输入和目标添加到model_inputs列表中，然后返回这个列表。
        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):  #这行代码遍历examples[prompt_column]列表的每一个元素。examples[prompt_column]表示从数据集中提取的问题或提示列。
            if examples[prompt_column][i] and examples[response_column][i]:  #检查第i个问题/提示和对应的回答是否存在。examples[prompt_column][i]和examples[response_column][i]分别表示第i个问题/提示和对应的回答。如果其中之一不存在，那么就跳过这个样本。
                query = examples[prompt_column][i]  #将第i个问题/提示赋值给变量query。
                history = examples[history_column][i] if history_column is not None else None  #检查是否存在历史对话列。如果存在，那么将第i个历史对话赋值给变量history；如果不存在，那么将None赋值给history。
                prompt = tokenizer.build_prompt(query, history)  #使用分词器的build_prompt函数将问题/提示和历史对话结合起来，生成模型的输入。这通常包括一些特定的格式和分词步骤。
                inputs.append(prompt)  #将生成的输入添加到inputs列表中。inputs列表将被用作模型的输入。
                targets.append(examples[response_column][i])  #将第i个回答添加到targets列表中。targets列表将被用作模型的目标。
                #在这段代码执行之后，你将获得两个列表：inputs和targets。inputs列表包含了所有的输入样本，targets列表包含了所有的目标样本。这两个列表将被用于模型的训练或评估。

        inputs = [prefix + inp for inp in inputs]  #对于输入列表inputs中的每个元素，都在它们的前面添加一个prefix，然后更新输入列表。这里的prefix可能是一个模型需要的特定前缀，比如特殊的开头标记。
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)  #使用tokenizer对更新后的输入进行处理，得到模型的输入。tokenizer是一个将原始文本转换为模型可以理解的形式的工具。这个处理包括截断和填充：如果输入的长度超过了data_args.max_source_length，则会被截断；如果输入的长度小于最大长度，则会被填充到最大长度。得到的model_inputs是一个字典，包含了输入的编码等信息。
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)  #对目标（也就是期望的输出）进行同样的处理，得到模型的标签。
 
        if data_args.ignore_pad_token_for_loss:  #如果设置了忽略填充标记的损失，则执行以下步骤：
            labels["input_ids"] = [  #对于标签中的每个输入ID，如果它是填充标记的ID，则将其替换为-100，否则保持不变。这是因为在计算损失时，我们通常希望忽略填充的部分。在PyTorch中，-100是一个特殊的值，表示在计算损失时忽略这个位置。
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]  #这行代码将处理后的标签添加到模型的输入中。这样，模型的输入就包含了输入和对应的标签，可以直接用于训练。

        return model_inputs

    #函数preprocess_function_train(examples)的主要目标是为模型训练阶段预处理数据。给定一些训练样例examples，它将为每个样例生成模型需要的输入和标签。这个过程包括以下几个步骤：
    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length + 1  #定义最大序列长度为源长度上限（即问题长度上限）加上目标长度上限（即答案长度上限）再加1。这个1通常是为特殊标记（比如序列结束标记）预留的空间。

        model_inputs = {  #初始化model_inputs字典，用于存储模型输入的数据。
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):  #遍历每一个样例。
            if examples[prompt_column][i] and examples[response_column][i]:  #如果样例的问题和答案都存在，那么处理这个样例。
                query, answer = examples[prompt_column][i], examples[response_column][i]  #获取问题和答案。

                history = examples[history_column][i] if history_column is not None else None  #获取历史对话，如果存在的话。
                prompt = tokenizer.build_prompt(query, history)  #用tokenizer.build_prompt方法来根据问题和历史对话构建提示。

                prompt = prefix + prompt  #在提示前面添加前缀。
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,  #将提示编码成模型可以理解的形式，得到输入ID序列a_ids。
                                         max_length=data_args.max_source_length)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,  #同样地，将答案编码成模型可以理解的形式，得到答案ID序列b_ids。
                                         max_length=data_args.max_target_length)

                context_length = len(a_ids)  #计算输入的长度。
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]  #将输入和答案的ID序列拼接起来，并在最后添加一个序列结束标记的ID，得到完整的输入序列。
                labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]  #标签序列的前context_length部分是填充标记的ID，后面是答案的ID序列和一个序列结束标记的ID。这样设置的原因是，我们只关心模型对答案部分的预测。
                
                pad_len = max_seq_length - len(input_ids)  #计算需要填充的长度。
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len  #在输入序列后面添加填充标记，使其长度达到max_seq_length。
                labels = labels + [tokenizer.pad_token_id] * pad_len  #同样地，也在标签序列后面添加填充标记。
                if data_args.ignore_pad_token_for_loss:  #如果设置了忽略填充标记的损失，那么将标签中的填充标记的ID替换为-100。
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]  #将处理好的输入和标签添加到model_inputs字典中。

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)  #在处理完所有样例后，返回model_inputs字典，它包含了所有样例的输入和标签，可以直接用于模型的训练。

        return model_inputs
    
    def print_dataset_example(example):
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.decode(example["labels"]))

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        print_dataset_example(train_dataset[0])

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0])

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_changed=model_args.pre_seq_len is not None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=max_seq_length, temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_seq_length, do_sample=True, top_p=0.7, temperature=0.95)
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p, l in zip(predictions, labels):
                        res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                        writer.write(f"{res}\n")
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
