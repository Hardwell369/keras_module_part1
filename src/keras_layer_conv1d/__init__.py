from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Convolution"
friendly_name = "Conv1D - Keras"
doc_url = "https://keras.io/api/layers/convolution_layers/convolution1d/"
cacheable = False

logger = structlog.get_logger()

ACTIVATIONS = [
    "softmax",
    "elu",
    "selu",
    "softplus",
    "softsign",
    "relu",
    "tanh",
    "sigmoid",
    "hard_sigmoid",
    "linear",
    "None",
]

INITIALIZERS = [
    "Zeros",
    "Ones",
    "Constant",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "VarianceScaling",
    "Orthogonal",
    "Identiy",
    "lecun_uniform",
    "lecun_normal",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
]

REGULARIZERS = [
    "l1",
    "l2",
    "l1_l2",
    "None",
]

CONSTRAINTS = [
    "max_norm",
    "non_neg",
    "unit_norm",
    "min_max_norm",
    "None"
]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/convolution_layers/convolution1d/

    # import keras

    init_params = dict(
        # filters,
        # kernel_size,
        # strides = 1,
        # padding = "valid",
        # data_format = None,
        # dilation_rate = 1,
        # groups = 1,
        # activation = None,
        # use_bias = True,
        # kernel_initializer = "glorot_uniform",
        # bias_initializer = "zeros",
        # kernel_regularizer = None,
        # bias_regularizer = None,
        # activity_regularizer = None,
        # kernel_constraint = None,
        # bias_constraint = None,
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
    filters: I.int("输出空间维度，即卷积核的数量）", min=1),  # type: ignore
    kernel_size: I.int("卷积窗口的大小"),  # type: ignore
    strides: I.int("卷积的步长长度") = 1,  # type: ignore
    padding: I.choice("填充模式", ["valid", "same", "causal"]) = "valid",  # type: ignore
    data_format: I.choice("输入的维度顺序", ["channels_last", "channels_first"]) = None,  # type: ignore
    dilation_rate: I.int("膨胀率") = 1,  # type: ignore
    groups: I.int("输入沿通道轴分组的数量") = 1,  # type: ignore
    activation: I.choice("激活函数", ACTIVATIONS) = "linear",  # type: ignore
    use_bias: I.bool("是否在输出中添加偏置") = True,  # type: ignore
    kernel_initializer: I.choice("卷积核的初始化方法", INITIALIZERS) = "glorot_uniform",  # type: ignore
    bias_initializer: I.choice("偏置向量的初始化方法", INITIALIZERS) = "zeros",  # type: ignore
    kernel_regularizer: I.choice("卷积核的正则化函数", REGULARIZERS) = "None",  # type: ignore
    bias_regularizer: I.choice("偏置向量的正则化函数", REGULARIZERS) = "None",  # type: ignore
    activity_regularizer: I.choice("输出的正则化函数", REGULARIZERS) = "None",  # type: ignore
    kernel_constraint: I.choice("应用于卷积核的约束函数", CONSTRAINTS) = "None",  # type: ignore
    bias_constraint: I.choice("应用于偏置向量的约束函数", CONSTRAINTS) = "None",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """一维卷积层（即时域卷积），用以在一维输入信号上进行邻域滤波。"""

    import keras

    init_params = dict(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=_none(kernel_regularizer),
        bias_regularizer=_none(bias_regularizer),
        activity_regularizer=_none(activity_regularizer),
        kernel_constraint=_none(kernel_constraint),
        bias_constraint=_none(bias_constraint),
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

    layer = keras.layers.Conv1D(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs
