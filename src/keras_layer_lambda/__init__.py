from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Core"
friendly_name = "Lambda - Keras"
doc_url = "https://keras.io/api/layers/core_layers/lambda/"
cacheable = False

logger = structlog.get_logger()

DEFAULT_FUNCTION = """def bigquant_run(x):
    # x为输入，即上一层的输出
    # 在这里添加您的代码
    return x + 1
"""

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/core_layers/lambda/

    # import keras

    init_params = dict(
        # function,
        # output_shape=None,
        # mask=None,
        # arguments=None,
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
    function: I.code("需要被评估的函数。接收输入张量作为第一个参数", language="python", specific_type_name="函数", default = DEFAULT_FUNCTION) = None,  # type: ignore
    output_shape: I.str("从函数中预期输出的形状。可以是元组或函数。如果是元组，它只指定第一个维度；如果是函数，它指定整个形状") = None,  # type: ignore
    mask: I.str("掩码，None表示不进行掩码操作") = None,  # type: ignore
    arguments: I.str("可选的关键字参数字典，将传递给函数", specific_type_name="dict") = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Lambda 是 Keras 中用于将任意表达式包装为层对象的类。这使得在构建 Sequential 和 Functional API 模型时可以使用任意表达式作为层。"""

    import keras

    init_params = dict(
        function=function,
        output_shape=eval(output_shape) if output_shape else None,
        mask=eval(mask) if mask else None,
        arguments=eval(arguments) if arguments else None,
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

    layer = keras.layers.Lambda(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs
