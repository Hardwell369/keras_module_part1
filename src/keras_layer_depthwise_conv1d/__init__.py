from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Convolution"
friendly_name = "DepthwiseConv1D - Keras"
doc_url = "https://keras.io/api/layers/convolution_layers/depthwise_convolution1d/"
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
    "L1L2",
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
    # https://keras.io/api/layers/convolution_layers/depthwise_convolution1d/

    # import keras

    init_params = dict(
        # kernel_size,
        # strides=1,
        # padding="valid",
        # depth_multiplier=1,
        # data_format=None,
        # dilation_rate=1,
        # activation=None,
        # use_bias=True,
        # depthwise_initializer="glorot_uniform",
        # bias_initializer="zeros",
        # depthwise_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # depthwise_constraint=None,
        # bias_constraint=None,
        # **kwargs,
    )
    call_params = dict(
        # inputs,
        # training=None,
        # mask=None,
    )

    return {"init": init_params, "call": call_params}
"""


def _none(x):
    if x == "None":
        return None
    return x


def run(
    kernel_size: I.str("卷积核大小，可以是一个整数或一个包含1个整数的元组/列表") = "3",  # type: ignore
    strides: I.str("卷积步长，可以是一个整数或一个包含1个整数的元组/列表") = "1",  # type: ignore
    padding: I.choice("填充方式，valid表示不填充，same表示填充使得输出尺寸等于输入尺寸", values=["valid", "same"]) = "valid",  # type: ignore
    depth_multiplier: I.int("每个输入通道的深度卷积输出通道数", min=1) = 1,  # type: ignore
    data_format: I.choice("数据格式，channels_last表示通道在最后，channels_first表示通道在最前", values=["channels_last", "channels_first"]) = "channels_last",  # type: ignore
    dilation_rate: I.str("膨胀率，可以是一个整数或一个包含1个整数的元组/列表") = "1",  # type: ignore
    activation: I.choice("激活函数", ACTIVATIONS) = "relu",  # type: ignore
    use_bias: I.bool("是否使用偏置项") = True,  # type: ignore
    depthwise_initializer: I.choice("深度卷积核初始化方法", INITIALIZERS) = "Zeros",  # type: ignore
    bias_initializer: I.choice("偏置向量初始化方法", INITIALIZERS) = "Zeros",  # type: ignore
    depthwise_regularizer: I.choice("深度卷积核正则项", REGULARIZERS) = "None",  # type: ignore
    bias_regularizer: I.choice("偏置向量正则项", REGULARIZERS) = "None",  # type: ignore
    activity_regularizer: I.choice("输出正则项", REGULARIZERS) = "None",  # type: ignore
    depthwise_constraint: I.choice("深度卷积核约束项", CONSTRAINTS) = "None",  # type: ignore
    bias_constraint: I.choice("偏置向量约束项", CONSTRAINTS) = "None",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Keras DepthwiseConv1D 层"""

    import keras

    def parse_tuple_or_int(value):
        try:
            if isinstance(value, str):
                value = eval(value)
            if isinstance(value, int):
                return value
            elif isinstance(value, tuple) or isinstance(value, list):
                return tuple(value)
            else:
                raise ValueError
        except:
            raise ValueError("参数格式不正确，应为整数或包含1个整数的元组/列表")

    init_params = dict(
        kernel_size=parse_tuple_or_int(kernel_size),
        strides=parse_tuple_or_int(strides),
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        dilation_rate=parse_tuple_or_int(dilation_rate),
        activation=_none(activation),
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=_none(depthwise_regularizer),
        bias_regularizer=_none(bias_regularizer),
        activity_regularizer=_none(activity_regularizer),
        depthwise_constraint=_none(depthwise_constraint),
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

    layer = keras.layers.DepthwiseConv1D(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs
