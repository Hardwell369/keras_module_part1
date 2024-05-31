from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Core"
friendly_name = "EinsumDense - Keras"
doc_url = "https://keras.io/api/layers/core_layers/einsum_dense/"
cacheable = False

logger = structlog.get_logger()

ACTIVATIONS = [
    "elu",
    "exponential",
    "gelu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "leaky_relu",
    "linear",
    "log_softmax",
    "mish",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "softsign",
    "swish",
    "tanh",
    "None",
]

INITIALIZERS = [
    "Constant",
    "GlorotNormal",
    "GlorotUniform",
    "HeNormal",
    "HeUniform",
    "Identity",
    "Initializer",
    "LecunNormal",
    "LecunUniform",
    "Ones",
    "OrthogonalInitializer",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "VarianceScaling",
    "Zeros",
    "None",
]

REGULARIZERS = ["L1", "L1L2", "L2", "OrthogonalRegularizer", "Regularizer", "None"]

CONSTRAINTS = ["Constraint", "MaxNorm", "MinMaxNorm", "NonNeg", "UnitNorm", "None"]

LORA_RANK_DESC = """lora_rank/低秩适应，这个参数用于实现 LoRA（Low-Rank Adaptation，低秩适应）。
LoRA 是一种用于减少大规模全连接层（Dense Layer）微调计算成本的方法。
LoRA（Low-Rank Adaptation）是一种技术，通过将层的权重矩阵分解为两个较低秩的可训练矩阵，从而减少参数数量和计算复杂度。这对于微调大型模型特别有用，因为它可以显著降低计算和存储成本，同时保持模型的性能。
如果设置了 lora_rank，EinsumDense 层的前向传递将实现 LoRA。具体来说，这会将层的权重矩阵设为不可训练，并用两个低秩可训练矩阵的乘积来表示这个权重矩阵的变化。这对于减少大型全连接层微调的计算成本非常有用。
你可以在创建 EinsumDense 层时通过设置 lora_rank 参数来启用 LoRA。如果你已经有一个定义好的 EinsumDense 层，也可以通过调用 layer.enable_lora(rank) 方法来启用 LoRA。
"""

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/core_layers/einsum_dense/

    # import keras

    init_params = dict(
        # equation,
        # output_shape,
        # activation=None,
        # bias_axes=None,
        # kernel_initializer="glorot_uniform",
        # bias_initializer="zeros",
        # kernel_regularizer=None,
        # bias_regularizer=None,
        # kernel_constraint=None,
        # bias_constraint=None,
        # lora_rank=None,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=True,
        # mask=None,
    )

    return {"init": init_params, "call": call_params}
"""


def _none(x):
    if x == "None":
        return None
    return x


def run(
    equation: I.str("描述要执行的einsum的方程。这必须是一个有效的einsum字符串。"),  # type: ignore
    output_shape: I.str("预期的输出张量形状（不包括批次维度和任何由省略号表示的维度）。对于任何未知或可以从输入形状推断的维度，可以指定None。"),  # type: ignore
    activation: I.choice("激活函数，用于将线性输出转换为非线性输出", ACTIVATIONS) = "linear",  # type: ignore
    bias_axes: I.str("包含要应用偏置的输出维度的字符串。") = None,  # type: ignore
    kernel_initializer: I.choice("权重矩阵初始化方法，kernel_initializer", INITIALIZERS) = "GlorotUniform",  # type: ignore
    bias_initializer: I.choice("偏置项初始化方法，bias_initializer，使用的是使用默认参数", INITIALIZERS) = "Zeros",  # type: ignore
    kernel_regularizer: I.choice("权重矩阵正则化方法，kernel_regularizer", REGULARIZERS) = "None",  # type: ignore
    bias_regularizer: I.choice("偏置项正则项，bias_regularizer", REGULARIZERS) = "None",  # type: ignore
    kernel_constraint: I.choice("权重矩阵上约束方法，kernel_constraint", CONSTRAINTS) = "None",  # type: ignore
    bias_constraint: I.choice("偏置项约束方法，bias_constraint", CONSTRAINTS) = "None",  # type: ignore
    lora_rank: I.int(LORA_RANK_DESC) = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """EinsumDense 是 Keras 中用于执行爱因斯坦求和操作的层。"""

    import keras

    init_params = dict(
        equation=equation,
        output_shape=eval(output_shape),
        activation=_none(activation),
        bias_axes=bias_axes,
        kernel_initializer=_none(kernel_initializer),
        bias_initializer=_none(bias_initializer),
        kernel_regularizer=_none(kernel_regularizer),
        bias_regularizer=_none(bias_regularizer),
        kernel_constraint=_none(kernel_constraint),
        bias_constraint=_none(bias_constraint),
        lora_rank=lora_rank,
    )
    call_params = dict()
    if input_layer is not None:
        call_params["inputs"] = input_layer
    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.EinsumDense(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs