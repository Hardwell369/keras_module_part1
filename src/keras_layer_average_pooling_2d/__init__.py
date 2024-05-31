from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Pooling"
friendly_name = "AveragePooling2D - Keras"
doc_url = "https://keras.io/api/layers/pooling_layers/average_pooling2d/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/pooling_layers/average_pooling2d/

    # import keras

    init_params = dict(
        # pool_size,
        # strides=None,
        # padding="valid",
        # data_format=None,
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
    pool_size: I.str("池化窗口大小，可以是一个整数或一个包含两个整数的元组"),  # type: ignore
    strides: I.str("池化步长，可以是一个整数或一个包含两个整数的元组，默认为池化窗口大小") = None,  # type: ignore
    padding: I.choice("填充模式", ["valid", "same"]) = "valid",  # type: ignore
    data_format: I.choice("数据格式", ["channels_last", "channels_first"]) = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Keras AveragePooling2D 层"""

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
            raise ValueError("参数格式不正确，应为整数或包含3个整数的元组/列表")

    init_params = dict(
        pool_size=parse_tuple_or_int(pool_size),
        strides=parse_tuple_or_int(strides) if strides else None,
        padding=padding,
        data_format=data_format,
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

    layer = keras.layers.AveragePooling2D(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs
