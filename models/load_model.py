from .braingnn import BrainGNN, BrainGNNRegression
from .bnt import BrainNetworkTransformer, BNTRegression
from .lggnn import LGGNN, LGGNNRegression
from .combraintf import ComBrainTF, ComBrainTFRegression
from .ibgnn import IBGNN, IBGNNRegression
from .brainnetcnn import BrainNetCNN, BrainNetCNNRegression, BrainNetCNNDeep, create_brainnetcnn


def load_model(model_name, hparams=None):
    #number of transformer stages
    n_stages = len(hparams.depths) if hasattr(hparams, 'depths') else 4

    if model_name == "neurostorm":
        # Lazy import: NeuroSTORM depends on mamba_ssm which may not be
        # installed in lightweight / CI environments. Other models should
        # still be loadable without it.
        from .neurostorm import NeuroSTORM, NeuroSTORMMAE
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
                time_mask=hparams.time_mask,
                atlas_map_path=getattr(hparams, 'atlas_map_path', None),
                use_strd=getattr(hparams, 'use_strd', False),
                strd_l_spat=getattr(hparams, 'strd_l_spat', 5),
                strd_l_temp=getattr(hparams, 'strd_l_temp', 5),
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
                attn_drop_rate=hparams.attn_drop_rate,
                prompt_len=(getattr(hparams, 'prompt_len', 0)
                            if getattr(hparams, 'use_prompt_tuning', False) else 0),
            )
    elif model_name == "swift":
        from .swift import SwiFT
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
    elif model_name == "braingnn":
        # BrainGNN for graph-based fMRI analysis
        num_rois = getattr(hparams, 'num_rois', 200)
        pooling_ratio = getattr(hparams, 'pooling_ratio', 0.5)
        num_communities = getattr(hparams, 'num_communities', 8)

        if hparams.downstream_task_type == 'regression':
            net = BrainGNNRegression(
                in_channels=num_rois,
                num_classes=1,
                num_rois=num_rois,
                pooling_ratio=pooling_ratio,
                num_communities=num_communities,
                dropout=getattr(hparams, 'dropout', 0.5)
            )
        else:
            net = BrainGNN(
                in_channels=num_rois,
                num_classes=hparams.num_classes,
                num_rois=num_rois,
                pooling_ratio=pooling_ratio,
                num_communities=num_communities,
                dropout=getattr(hparams, 'dropout', 0.5)
            )
    elif model_name == "bnt":
        # BrainNetworkTransformer for FC-based fMRI analysis
        num_rois = getattr(hparams, 'num_rois', 200)
        pos_encoding = getattr(hparams, 'pos_encoding', 'identity')
        pos_embed_dim = getattr(hparams, 'pos_embed_dim', 8)
        pooling_sizes = getattr(hparams, 'pooling_sizes', [100, 50, 25])
        do_pooling = getattr(hparams, 'do_pooling', [True, True, False])
        hidden_size = getattr(hparams, 'hidden_size', 1024)

        # Auto-scale pooling sizes when using defaults that don't match num_rois
        if pooling_sizes == [100, 50, 25]:
            pooling_sizes = [num_rois // 2, num_rois // 10]
            do_pooling = [True, True]

        if hparams.downstream_task_type == 'regression':
            net = BNTRegression(
                num_rois=num_rois,
                node_feature_size=num_rois,
                num_classes=1,
                pos_encoding=pos_encoding,
                pos_embed_dim=pos_embed_dim,
                pooling_sizes=pooling_sizes,
                do_pooling=do_pooling,
                hidden_size=hidden_size,
                dropout=getattr(hparams, 'dropout', 0.1)
            )
        else:
            net = BrainNetworkTransformer(
                num_rois=num_rois,
                node_feature_size=num_rois,
                num_classes=hparams.num_classes,
                pos_encoding=pos_encoding,
                pos_embed_dim=pos_embed_dim,
                pooling_sizes=pooling_sizes,
                do_pooling=do_pooling,
                hidden_size=hidden_size,
                dropout=getattr(hparams, 'dropout', 0.1)
            )
    elif model_name == "lggnn":
        # LG-GNN for learnable graph structure
        num_rois = getattr(hparams, 'num_rois', 200)
        hidden_dims = getattr(hparams, 'hidden_dims', [128, 64])
        k_neighbors = getattr(hparams, 'k_neighbors', 10)
        learn_graph = getattr(hparams, 'learn_graph', True)
        graph_metric = getattr(hparams, 'graph_metric', 'cosine')

        if hparams.downstream_task_type == 'regression':
            net = LGGNNRegression(
                num_rois=num_rois,
                node_feature_dim=num_rois,
                hidden_dims=hidden_dims,
                num_classes=1,
                k_neighbors=k_neighbors,
                learn_graph=learn_graph,
                graph_metric=graph_metric,
                dropout=getattr(hparams, 'dropout', 0.5)
            )
        else:
            net = LGGNN(
                num_rois=num_rois,
                node_feature_dim=num_rois,
                hidden_dims=hidden_dims,
                num_classes=hparams.num_classes,
                k_neighbors=k_neighbors,
                learn_graph=learn_graph,
                graph_metric=graph_metric,
                dropout=getattr(hparams, 'dropout', 0.5)
            )
    elif model_name == "combraintf":
        # Com-BrainTF for community-aware brain analysis
        num_rois = getattr(hparams, 'num_rois', 200)
        num_communities = getattr(hparams, 'num_communities', 10)
        d_model = getattr(hparams, 'd_model', 128)
        nhead = getattr(hparams, 'nhead', 4)
        num_layers = getattr(hparams, 'num_layers', 3)
        dim_feedforward = getattr(hparams, 'dim_feedforward', 512)
        use_community_mask = getattr(hparams, 'use_community_mask', True)

        if hparams.downstream_task_type == 'regression':
            net = ComBrainTFRegression(
                num_rois=num_rois,
                node_feature_dim=num_rois,
                num_communities=num_communities,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                num_classes=1,
                dropout=getattr(hparams, 'dropout', 0.1),
                use_community_mask=use_community_mask
            )
        else:
            net = ComBrainTF(
                num_rois=num_rois,
                node_feature_dim=num_rois,
                num_communities=num_communities,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                num_classes=hparams.num_classes,
                dropout=getattr(hparams, 'dropout', 0.1),
                use_community_mask=use_community_mask
            )
    elif model_name == "ibgnn":
        # IBGNN for interpretable brain graph analysis
        num_rois = getattr(hparams, 'num_rois', 200)
        hidden_dims = getattr(hparams, 'hidden_dims', [128, 64])
        use_edge_attr = getattr(hparams, 'use_edge_attr', True)

        if hparams.downstream_task_type == 'regression':
            net = IBGNNRegression(
                num_rois=num_rois,
                node_feature_dim=num_rois,
                hidden_dims=hidden_dims,
                num_classes=1,
                dropout=getattr(hparams, 'dropout', 0.5),
                use_edge_attr=use_edge_attr
            )
        else:
            net = IBGNN(
                num_rois=num_rois,
                node_feature_dim=num_rois,
                hidden_dims=hidden_dims,
                num_classes=hparams.num_classes,
                dropout=getattr(hparams, 'dropout', 0.5),
                use_edge_attr=use_edge_attr
            )
    elif model_name == "brainnetcnn":
        num_rois = getattr(hparams, 'num_rois', 200)
        variant = getattr(hparams, 'brainnetcnn_variant', 'standard')
        e2e_channels = getattr(hparams, 'e2e_channels', [32])
        e2n_channels = getattr(hparams, 'e2n_channels', 64)
        n2g_channels = getattr(hparams, 'n2g_channels', 256)

        num_classes = 1 if hparams.downstream_task_type == 'regression' else hparams.num_classes
        task_type = 'regression' if hparams.downstream_task_type == 'regression' else 'classification'
        net = create_brainnetcnn(
            task_type=task_type, variant=variant,
            num_rois=num_rois, num_classes=num_classes,
            e2e_channels=e2e_channels, e2n_channels=e2n_channels,
            n2g_channels=n2g_channels, dropout=getattr(hparams, 'dropout', 0.5)
        )
    elif model_name == "emb_mlp":
        from .heads import EmbHead
        num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1))
        net = EmbHead(final_embedding_size=128, num_tokens=num_tokens, use_normalization=True)
    elif model_name == "clf_mlp":
        from .heads import ClsHead
        num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1))
        net = ClsHead(version=hparams.clf_head_version, num_classes=hparams.num_classes, num_tokens=num_tokens)
    elif model_name == "reg_mlp":
        from .heads import RegHead
        num_tokens = hparams.embed_dim * (hparams.c_multiplier ** (n_stages - 1))
        net = RegHead(version=1, num_tokens=num_tokens)
    else:
        raise NameError(f"{model_name} is a wrong model name")

    return net
