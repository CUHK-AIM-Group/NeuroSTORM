"""
Test model loading and initialization for all NeuroSTORM models.
"""

import pytest
import torch


class TestModelLoading:
    """Test that all models can be imported and initialized."""

    def test_braingnn_import(self):
        """Test BrainGNN import."""
        from models.braingnn import BrainGNN, BrainGNNRegression
        assert BrainGNN is not None
        assert BrainGNNRegression is not None

    def test_bnt_import(self):
        """Test BNT import."""
        from models.bnt import BrainNetworkTransformer, BNTRegression
        assert BrainNetworkTransformer is not None
        assert BNTRegression is not None

    def test_lggnn_import(self):
        """Test LG-GNN import."""
        from models.lggnn import LGGNN, LGGNNRegression
        assert LGGNN is not None
        assert LGGNNRegression is not None

    def test_combraintf_import(self):
        """Test Com-BrainTF import."""
        from models.combraintf import ComBrainTF, ComBrainTFRegression
        assert ComBrainTF is not None
        assert ComBrainTFRegression is not None

    def test_ibgnn_import(self):
        """Test IBGNN import."""
        from models.ibgnn import IBGNN, IBGNNRegression
        assert IBGNN is not None
        assert IBGNNRegression is not None

    def test_brainnetcnn_import(self):
        """Test BrainNetCNN import."""
        from models.brainnetcnn import BrainNetCNN, BrainNetCNNRegression
        assert BrainNetCNN is not None
        assert BrainNetCNNRegression is not None

    @pytest.mark.unit
    def test_bnt_initialization(self):
        """Test BNT model initialization."""
        from models.bnt import BrainNetworkTransformer

        model = BrainNetworkTransformer(
            num_rois=200,
            node_feature_size=200,
            num_classes=2,
            pooling_sizes=[100, 50, 25],
            do_pooling=[True, True, False]
        )
        assert model is not None
        assert model.num_rois == 200

    @pytest.mark.unit
    def test_brainnetcnn_initialization(self):
        """Test BrainNetCNN model initialization."""
        from models.brainnetcnn import BrainNetCNN

        model = BrainNetCNN(
            num_rois=200,
            num_classes=2
        )
        assert model is not None
        assert model.num_rois == 200

    @pytest.mark.unit
    def test_combraintf_initialization(self):
        """Test Com-BrainTF model initialization."""
        from models.combraintf import ComBrainTF

        model = ComBrainTF(
            num_rois=200,
            node_feature_dim=200,
            num_communities=10,
            num_classes=2
        )
        assert model is not None
        assert model.num_rois == 200
        assert model.num_communities == 10

    @pytest.mark.unit
    def test_bnt_forward_pass(self):
        """Test BNT forward pass."""
        from models.bnt import BrainNetworkTransformer

        model = BrainNetworkTransformer(
            num_rois=200,
            node_feature_size=200,
            num_classes=2,
            pooling_sizes=[100, 50, 25],
            do_pooling=[True, True, False]
        )
        model.eval()

        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 200, 200)

        with torch.no_grad():
            output, assignments = model(x)

        assert output.shape == (batch_size, 2)
        assert len(assignments) == 3

    @pytest.mark.unit
    def test_brainnetcnn_forward_pass(self):
        """Test BrainNetCNN forward pass."""
        from models.brainnetcnn import BrainNetCNN

        model = BrainNetCNN(num_rois=200, num_classes=2)
        model.eval()

        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 200, 200)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 2)

    @pytest.mark.unit
    def test_combraintf_forward_pass(self):
        """Test Com-BrainTF forward pass."""
        from models.combraintf import ComBrainTF

        model = ComBrainTF(
            num_rois=200,
            node_feature_dim=200,
            num_communities=10,
            num_classes=2
        )
        model.eval()

        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 200, 200)

        with torch.no_grad():
            output, assignment = model(x)

        assert output.shape == (batch_size, 2)
        assert assignment.shape == (200, 10)


class TestModelRegistry:
    """Test model loading through load_model function."""

    def test_load_model_function(self):
        """Test load_model function exists."""
        from models.load_model import load_model
        assert load_model is not None

    @pytest.mark.unit
    def test_load_bnt(self):
        """Test loading BNT through load_model."""
        from models.load_model import load_model
        from argparse import Namespace

        hparams = Namespace(
            num_rois=200,
            pos_encoding='identity',
            pos_embed_dim=32,
            pooling_sizes=[100, 50, 25],
            do_pooling=[True, True, False],
            hidden_size=1024,
            dropout=0.1,
            num_classes=2,
            downstream_task_type='classification'
        )

        model = load_model('bnt', hparams)
        assert model is not None

    @pytest.mark.unit
    def test_load_brainnetcnn(self):
        """Test loading BrainNetCNN through load_model."""
        from models.load_model import load_model
        from argparse import Namespace

        hparams = Namespace(
            num_rois=200,
            brainnetcnn_variant='standard',
            e2e_channels=[32, 64, 64],
            e2n_channels=128,
            n2g_channels=256,
            fc_channels=[128, 64],
            dropout=0.5,
            num_classes=2,
            downstream_task_type='classification'
        )

        model = load_model('brainnetcnn', hparams)
        assert model is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
