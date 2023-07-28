# 导入transformers库的PretrainedConfig模块
from transformers import PretrainedConfig

# 定义一个名为ChatGLMConfig的新类，继承自PretrainedConfig
class ChatGLMConfig(PretrainedConfig):
    # 定义模型的类型为"chatglm"
    model_type = "chatglm"

    # 定义类的初始化函数，设置模型的各种配置参数和默认值
    def __init__(
        self,
        num_layers=28,  # 定义模型中的层数，默认为28
        padded_vocab_size=65024,  # 定义词汇表的大小，默认为65024
        hidden_size=4096,  # 定义隐藏层的大小，默认为4096
        ffn_hidden_size=13696,  # 定义前馈神经网络的隐藏层大小，默认为13696
        kv_channels=128,  # 定义键值对的通道数量，默认为128
        num_attention_heads=32,  # 定义注意力头的数量，默认为32
        seq_length=2048,  # 定义序列长度，默认为2048
        hidden_dropout=0.0,  # 定义隐藏层的dropout比例，默认为0
        attention_dropout=0.0,  # 定义注意力层的dropout比例，默认为0
        layernorm_epsilon=1e-5,  # 定义LayerNorm层中的一个小常数，默认为1e-5
        rmsnorm=True,  # 定义是否使用RMS Normalization，默认为True
        apply_residual_connection_post_layernorm=False,  # 定义是否在LayerNorm后应用残差连接，默认为False
        post_layer_norm=True,  # 定义是否应用Post-Layer Norm，默认为True
        add_bias_linear=False,  # 定义是否在线性层添加偏置项，默认为False
        add_qkv_bias=False,  # 定义是否在查询/键/值三个权重矩阵上添加偏置项，默认为False
        bias_dropout_fusion=True,  # 定义是否将偏置和dropout融合，默认为True
        multi_query_attention=False,  # 定义是否使用多查询注意力，默认为False
        multi_query_group_num=1,  # 定义多查询组的数量，默认为1
        apply_query_key_layer_scaling=True,  # 定义是否应用查询键层的缩放，默认为True
        attention_softmax_in_fp32=True,  # 定义注意力softmax是否使用单精度浮点数，默认为True
        fp32_residual_connection=False,  # 定义是否在残差连接中使用单精度浮点数，默认为False
        quantization_bit=0,  # 定义量化位数，默认为0
        pre_seq_len=None,  # 定义预序列长度，默认为None
        prefix_projection=False,  # 定义是否应用前缀投影，默认为False
        **kwargs  # 接收其他以关键字方式给出的参数
    ):
        
        
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)
