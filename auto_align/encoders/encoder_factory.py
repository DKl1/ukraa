from auto_align.constants.languages_pairs_encoder_models import ENCODER_MODELS


def get_encoder(src_lang: str, tgt_lang: str):
    """
    Return the appropriate encoder based on the language pair.
    """
    pair_key = f"{src_lang}-{tgt_lang}"
    model_name = ENCODER_MODELS.get(pair_key, "LaBSE")

    if model_name == "LaBSE":
        from .labse_encoder import LaBSEEncoder
        return LaBSEEncoder()
    elif model_name == "LASER":
        from .laser_encoder_custom import LaserEncoder
        return LaserEncoder()
    elif model_name == "SBERT":
        from .sbert_encoder import SBERTEncoder
        return SBERTEncoder()
    else:
        raise ValueError(f"Unsupported encoder model: {model_name}")
