from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Core"
friendly_name = "Masking - Keras"
doc_url = "https://keras.io/api/layers/core_layers/masking/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/core_layers/masking/

    # import keras

    init_params = dict(
        # mask_value = 0.0,
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
    mask_value: I.float("用于跳过时间步的掩码值") = 0.0,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """屏蔽层。使用给定的值对输入的序列信号进行“屏蔽”，用以定位需要跳过的时间步。对于输入张量的时间步，即输入张量的第1维度（维度从0开始算，见例子），如果输入张量在该时间步上都等于mask_value，则该时间步将在模型接下来的所有层（只要支持masking）被跳过（屏蔽）。如果模型接下来的一些层不支持masking，却接受到masking过的数据，则抛出异常。"""

    import keras

    init_params = dict(
        mask_value=mask_value,
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

    layer = keras.layers.Masking(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs
