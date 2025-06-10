from .neurostorm import NeuroSTORM, NeuroSTORMMAE
from .swift import SwiFT


def load_model(model_name, hparams=None):
    #number of transformer stages
    n_stages = len(hparams.depths)

    if model_name == "neurostorm":
        if hparams.pretraining:
            net = NeuroSTORMMAE(
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
                drop_rate=hparams.attn_drop_rate,
                drop_path_rate=hparams.attn_drop_rate,
                attn_drop_rate=hparams.attn_drop_rate,
                mask_ratio=hparams.mask_ratio,
                spatial_mask=hparams.spatial_mask,
                time_mask=hparams.time_mask
            )
        else:
            net = NeuroSTORM(
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
                drop_rate=hparams.attn_drop_rate,
                drop_path_rate=hparams.attn_drop_rate,
                attn_drop_rate=hparams.attn_drop_rate
            )
    elif model_name == "swift":
        net = SwiFT(
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
            drop_rate=hparams.attn_drop_rate,
            drop_path_rate=hparams.attn_drop_rate,
            attn_drop_rate=hparams.attn_drop_rate
        )
    elif model_name == "emb_mlp":
        from .heads.emb_head import emb_head
        net = emb_head(final_embedding_size=128, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)), use_normalization=True)
    elif model_name == "clf_mlp":
        from .heads.cls_head import cls_head
        net = cls_head(version=hparams.clf_head_version, num_classes=hparams.num_classes, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)))
    elif model_name == "reg_mlp":
        from .heads.reg_head import reg_head
        net = reg_head(version=1, num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1)))
    else:
        raise NameError(f"{model_name} is a wrong model name")

    return net
