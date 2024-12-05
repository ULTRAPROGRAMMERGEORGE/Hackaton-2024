import torch

from timm.models.vision_transformer import VisionTransformer


def moonshot(model_filepath: str) -> VisionTransformer:
    """Создает модель (архитектура vit-large-patch14-reg4-518)
    и подгружает предобученные веса Moonshot.
    """
    model = VisionTransformer(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        reg_tokens=4,
        no_embed_class=True
    )

    model.load_state_dict(torch.load(model_filepath))

    return model
