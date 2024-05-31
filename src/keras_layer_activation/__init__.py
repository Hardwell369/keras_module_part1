from bigmodule import I

import structlog
# metadata
# 模块作者
author = "BigQuant"
# 模块分类
category = r"深度学习\Keras\Core"
# 模块显示名
friendly_name = "Activation - Keras"
# 文档地址, optional
doc_url = "https://keras.io/api/layers/core_layers/activation/"
# 是否自动缓存结果
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

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/core_layers/activation/

    # import keras

    init_params = dict(
        # activation = "relu",
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=True,
        # mask=None,
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    activation: I.choice('激活函数', ACTIVATIONS) = 'tanh',  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port('输入') = None  # type: ignore
)->[
    I.port('输出', 'data')  # type: ignore
]:
    """Keras Activation 层"""

    import keras

    init_params = dict(
        activation=activation
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

    layer = keras.layers.Activation(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs
