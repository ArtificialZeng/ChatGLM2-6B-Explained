PRE_SEQ_LEN=128  #预设定的序列长度，表示在模型输入中，将处理的最大文本序列长度设置为128个词或字符。
LR=2e-2  #LR是学习率（Learning Rate）的简写，值为0.02。学习率是优化算法的一个超参数，控制模型在训练过程中学习的速度。过大的学习率可能会导致训练收敛的不稳定，过小的学习率可能会导致训练过程过于缓慢。
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \  #是使用torchrun来运行一个分布式程序的命令。这里指定了单节点（nnodes=1）运行，并在每个节点上运行的进程数为设定的GPU数量。
    --do_train \  #是一个标志位，指示该程序应执行训练过程。
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --preprocessing_num_workers 10 \  #这个参数指定了预处理阶段并行工作的线程数量。
    --prompt_column content \
    --response_column summary \  #这些参数定义了在训练和验证数据中，模型输入的列名（prompt_column）以及模型数据回答的列名（response_column）。
    --overwrite_cache \  #一个标志位，如果设置，那么在加载数据前将删除预处理的缓存。
    --model_name_or_path THUDM/chatglm2-6b \  #这个参数指定了预训练模型的名称或者路径，模型将在这个预训练模型的基础上进行微调。
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \  #一个标志位，如果设置，那么在训练开始时将删除输出目录，以便重新开始训练。
    --max_source_length 64 \
    --max_target_length 128 \  # 这些参数定义了源输入和目标输出的最大长度。
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \  #参数定义了每个设备（即GPU）的训练和评估批次大小。
    --gradient_accumulation_steps 16 \  #这个参数定义了在进行一次参数更新之前，需要进行的梯度累积步骤数量。这是一种内存优化策略，可以使得在内存受限的情况下训练更大的模型。
    --predict_with_generate \  #一个标志位，如果设置，那么将使用生成式的方法（例如，自回归解码）来进行预测。
    --max_steps 3000 \  #定义了训练过程的最大步数。
    --logging_steps 10 \ #定义了记录日志和保存模型的步数间隔。
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \  #这些参数在上面已经定义过了，这里是将它们应用于训练过程。
    --quantization_bit 4  #定义了模型权重量化的位数。使用模型量化可以减少模型的存储需求，并可能提高推理速度，但可能会以精度为代价。这里设定为4位，意味着每个模型权重值都将映射到16（2的4次方）个不同的值。

