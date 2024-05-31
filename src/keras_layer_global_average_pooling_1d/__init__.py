from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Pooling"
friendly_name = "GlobalAveragePooling1D - Keras"
doc_url = "https://keras.io/api/layers/pooling_layers/global_average_pooling1d/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/pooling_layers/global_average_pooling1d/

    # import keras

    init_params = dict(
        # data_format=None,
        # keepdims=False,
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
    data_format: I.choice("数据格式", ["channels_last", "channels_first"]) = "channels_last",  # type: ignore
    keepdims: I.bool("是否保留时间维度") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入层", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Keras GlobalAveragePooling1D 层"""

    import keras

    init_params = dict(
        data_format=data_format,
        keepdims=keepdims,
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

    layer = keras.layers.GlobalAveragePooling1D(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs
