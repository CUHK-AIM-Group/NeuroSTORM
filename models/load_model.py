from .swift import SwinTransformer4D as SwinTransformer4D_ver7
from .swift import SwinTransformer4DMAE

def load_model(model_name, hparams=None):
    #number of transformer stages
    n_stages = len(hparams.depths)

    if hparams.precision == 16:
        to_float = False
    elif hparams.precision == 32:
        to_float = True

    print(to_float)

    if model_name == "swin4d_ver7":
        net = SwinTransformer4D_ver7(
            img_size=hparams.img_size,
            in_chans=hparams.in_chans,
            embed_dim=hparams.embed_dim,
            window_size=hparams.window_size,
            first_window_size=hparams.first_window_size,
            patch_size=hparams.patch_size,
            depths=hparams.depths,
            num_heads=hparams.num_heads,
            c_multiplier=hparams.c_multiplier,
            last_layer_full_MSA=hparams.last_layer_full_MSA,
            to_float=to_float,
            drop_rate=hparams.attn_drop_rate,
            drop_path_rate=hparams.attn_drop_rate,
            attn_drop_rate=hparams.attn_drop_rate,
            use_mamba=hparams.use_mamba,
        )
    elif model_name == "swin4d_mae":
        net = SwinTransformer4DMAE(
            img_size=hparams.img_size,
            in_chans=hparams.in_chans,
            embed_dim=hparams.embed_dim,
            window_size=hparams.window_size,
            first_window_size=hparams.first_window_size,
            patch_size=hparams.patch_size,
            depths=hparams.depths,
            num_heads=hparams.num_heads,
            c_multiplier=hparams.c_multiplier,
            last_layer_full_MSA=hparams.last_layer_full_MSA,
            to_float=to_float,
            drop_rate=hparams.attn_drop_rate,
            drop_path_rate=hparams.attn_drop_rate,
            attn_drop_rate=hparams.attn_drop_rate,
            mask_ratio=hparams.mask_ratio,
            spatial_mask=hparams.spatial_mask,
            time_mask=hparams.time_mask,
            use_mamba=hparams.use_mamba,
        )
    elif model_name == "emb_mlp":
        from .emb_mlp import mlp
        net = mlp(final_embedding_size=128, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)), use_normalization=True)
    elif model_name == "clf_mlp":
        if hparams.clf_head_version == 'v1':
            from .clf_mlp import mlp
            net = mlp(num_classes=hparams.num_classes, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)))
        elif hparams.clf_head_version == 'v2':
            from .clf_mlp_v2 import mlp
            net = mlp(num_classes=hparams.num_classes, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)))
        elif hparams.clf_head_version == 'v3':
            from .clf_mlp_v3 import ViTClassifier
            net = ViTClassifier(num_classes=hparams.num_classes, emb_size = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)))
        else:
            raise NotImplementedError
    elif model_name == "reg_mlp":
        from .clf_mlp import mlp
        net = mlp(num_classes=1, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)))
    else:
        raise NameError(f"{model_name} is a wrong model name")

    return net
