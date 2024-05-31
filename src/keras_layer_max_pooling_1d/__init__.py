from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Pooling"
friendly_name = "MaxPooling1D - Keras"
doc_url = "https://keras.io/api/layers/pooling_layers/max_pooling1d/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/pooling_layers/max_pooling1d/

    # import keras

    init_params = dict(
        # pool_size=2,
        # strides=None,
        # padding="valid",
        # data_format="channels_last",
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
    pool_size: I.int("Max pooling 窗口大小, pool_size, 正整数", min=1) = 2,  # type: ignore
    strides: I.int("Max pooling 步幅, strides, 可以为正整数或 None。如果为 None，默认等于 pool_size") = None,  # type: ignore
    padding: I.choice("填充方式, padding，'valid' 表示不填充，'same' 表示填充保持输出大小等于输入大小", ["valid", "same"]) = "valid",  # type: ignore
    data_format: I.choice("数据格式, data_format，输入数据的维度顺序，'channels_last' 表示 (batch, steps, features)，'channels_first' 表示 (batch, features, steps)", ["channels_last", "channels_first"]) = "channels_last",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Keras MaxPooling1D 层"""

    import keras

    init_params = dict(
        pool_size=pool_size,
        strides=strides,
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

    layer = keras.layers.MaxPooling1D(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs
