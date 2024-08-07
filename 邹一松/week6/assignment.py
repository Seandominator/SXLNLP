def parameter_calculation():
    """
    计算BERT模型参数
    """
    hidden_size = 522
    intermediate_size = 3072  # 4 *H
    max_position_embeddings = 512 
    num_hidden_layers = 12
    type_vocab_size = 2 
    vocab_size = 15532

    # 1、embedding 层参数 token + segment + position
    embedding = (vocab_size + max_position_embeddings + type_vocab_size) * hidden_size
    qkv_parameter = hidden_size * hidden_size * 3
    # 先过 softmax 函数，attention(Q，K，V)=softmax(QKT/d)QV) ; 然后过线性层 output=Liner(Attention(Q,K,V))
    self_output = (hidden_size * hidden_size) * 2  # 参数 W + B
    self_parameter = qkv_parameter + self_output

    # 3、BERT结构-FFN 层参数
    # FFN = output=Liner(gelu(Liner(x)))
    # L1 = WX +B ：X= H*4H B = H*4H
    # L2 = WX +B ：X= 4H*H B = 4H*H
    ffn_parameter = (hidden_size * intermediate_size) * 2 + (intermediate_size * hidden_size) * 2

    # 4、LayerNorm层: embedding、self-attention、FFN 之后
    # y=self.gamma * （x-u）/sqrt(exp(σ,2)+ϵ) +self.beta 其中 gamma 和 beta 是可训练的参数 维度为 1 * hidden_size
    # layer_norm = gamma + beta
    embedding_layer_norm = hidden_size * 2
    self_attention_layer_norm = hidden_size * 2
    ffn_layer_norm = hidden_size * 2

    # BaseBert包含 单层 embedding 层 + 12层encode 层

    # 最终结果如下：
    all_parameters = ((embedding + embedding_layer_norm) +
                      num_hidden_layers * (self_parameter + ffn_parameter + self_attention_layer_norm + ffn_layer_norm))

    print(all_parameters)
parameter_calculation()
