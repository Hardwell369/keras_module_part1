from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Core"
friendly_name = "Embedding - Keras"
doc_url = "https://keras.io/api/layers/core_layers/embedding/"
cacheable = False

logger = structlog.get_logger()

EMBEDDINGS_INITIALIZERS = [
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
    "uniform",
]

EMBEDDINGS_REGULARIZERS = ["L1", "L2", "L1L2", "None"]

EMBEDDINGS_CONSTRAINTS = ["max_norm", "non_neg", "unit_norm", "min_max_norm", "None"]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/core_layers/embedding/

    # import keras

    init_params = dict(
        # "input_dim": 1000,
        # "output_dim": 64,
        # "embeddings_initializer": "uniform",
        # "embeddings_regularizer": "None",
        # "embeddings_constraint": "None",
        # "mask_zero": False,
        # "weights": None,
        # "lora_rank": None,
        # "input_length": None,
        # "input_layer": None,
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
    input_dim: I.int("词汇表大小，即最大整数索引 + 1"),  # type: ignore
    output_dim: I.int("密集嵌入的维度"),  # type: ignore
    embeddings_initializer: I.choice("嵌入矩阵的初始化方法", EMBEDDINGS_INITIALIZERS) = "uniform",  # type: ignore
    embeddings_regularizer: I.choice("应用于嵌入矩阵的正则化函数", EMBEDDINGS_REGULARIZERS) = "None",  # type: ignore
    embeddings_constraint: I.choice("应用于嵌入矩阵的约束函数", EMBEDDINGS_CONSTRAINTS) = "None",  # type: ignore
    mask_zero: I.bool("输入值0是否是一个特殊的“填充”值，应被屏蔽") = False,  # type: ignore
    weights: I.port("初始嵌入值的可选浮点矩阵", optional=True) = None,  # type: ignore
    lora_rank: I.int("lora_rank, Optional integer. If set, the layer's forward pass will implement LoRA (Low-Rank Adaptation) with the provided rank.") = None,  # type: ignore
    input_length: I.int('input_length，当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断') = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Keras Embedding 层"""

    import keras

    init_params = dict(
        input_dim=input_dim,
        output_dim=output_dim,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=_none(embeddings_regularizer),
        embeddings_constraint=_none(embeddings_constraint),
        mask_zero=mask_zero,
        weights=weights,
        lora_rank=lora_rank,
        input_length=input_length,
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
        logger.info("Embedding", init_params=init_params, call_params=call_params)

    layer = keras.layers.Embedding(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs