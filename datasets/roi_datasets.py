"""
ROI-based and Functional Connectivity Dataset Loaders for NeuroSTORM

Supports:
1. ROI time series data (num_rois, num_timepoints)
2. Functional connectivity matrices (num_rois, num_rois)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_array
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce


class ROIDataset(Dataset):
    """
    Dataset for ROI time series data.

    Data format: .npy files with shape (num_rois, num_timepoints)
    """

    def __init__(
        self,
        root,
        subject_dict,
        atlas_name='cc200',
        sequence_length=None,
        stride=1,
        use_augmentations=False,
        train=True,
        **kwargs
    ):
        """
        Args:
            root: Root directory containing ROI data
            subject_dict: Dictionary mapping subject IDs to (sex, target) tuples
            atlas_name: Name of brain atlas used (e.g., 'cc200', 'aal', 'schaefer')
            sequence_length: Length of time series to use (None = use all)
            stride: Stride for sliding window
            use_augmentations: Whether to apply data augmentation
            train: Whether this is training set
        """
        super().__init__()
        self.root = root
        self.subject_dict = subject_dict
        self.atlas_name = atlas_name
        self.sequence_length = sequence_length
        self.stride = stride
        self.use_augmentations = use_augmentations
        self.train = train

        self.data = self._set_data()

    def _set_data(self):
        """Build list of (subject_id, subject_path, target, sex) tuples."""
        data = []
        roi_root = os.path.join(self.root, 'roi', self.atlas_name)

        for subject_id in self.subject_dict:
            sex, target = self.subject_dict[subject_id]

            # Look for ROI file
            roi_file = os.path.join(roi_root, f"{subject_id}_{self.atlas_name}.npy")

            if os.path.exists(roi_file):
                data.append((subject_id, roi_file, target, sex))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subject_id, roi_file, target, sex = self.data[idx]

        # Load ROI time series: (num_rois, num_timepoints)
        roi_data = np.load(roi_file)

        # Handle sequence length
        if self.sequence_length is not None and roi_data.shape[1] > self.sequence_length:
            # Take first sequence_length timepoints
            roi_data = roi_data[:, :self.sequence_length]

        # Convert to tensor: (num_rois, num_timepoints)
        roi_tensor = torch.from_numpy(roi_data).float()

        return {
            "roi_sequence": roi_tensor,
            "subject_name": subject_id,
            "target": target,
            "sex": sex,
        }


class FCDataset(Dataset):
    """
    Dataset for functional connectivity matrices.

    Data format: .npy files with shape (num_rois, num_rois)
    Can be computed from ROI time series using correlation/partial correlation.
    """

    def __init__(
        self,
        root,
        subject_dict,
        atlas_name='cc200',
        fc_type='correlation',  # 'correlation' or 'partial_correlation'
        use_augmentations=False,
        train=True,
        return_format='matrix',  # 'matrix' or 'bnt' (for BrainNetworkTransformer)
        **kwargs
    ):
        """
        Args:
            root: Root directory containing FC data
            subject_dict: Dictionary mapping subject IDs to (sex, target) tuples
            atlas_name: Name of brain atlas used
            fc_type: Type of functional connectivity ('correlation' or 'partial_correlation')
            use_augmentations: Whether to apply data augmentation
            train: Whether this is training set
            return_format: 'matrix' returns (num_rois, num_rois), 'bnt' returns (num_rois, num_rois) for BNT input
        """
        super().__init__()
        self.root = root
        self.subject_dict = subject_dict
        self.atlas_name = atlas_name
        self.fc_type = fc_type
        self.use_augmentations = use_augmentations
        self.train = train
        self.return_format = return_format

        self.data = self._set_data()

    def _set_data(self):
        """Build list of (subject_id, fc_path, target, sex) tuples."""
        data = []
        fc_root = os.path.join(self.root, 'fc', self.atlas_name, self.fc_type)

        for subject_id in self.subject_dict:
            sex, target = self.subject_dict[subject_id]

            # Look for FC file
            fc_file = os.path.join(fc_root, f"{subject_id}_{self.atlas_name}_{self.fc_type}.npy")

            if os.path.exists(fc_file):
                data.append((subject_id, fc_file, target, sex))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subject_id, fc_file, target, sex = self.data[idx]

        # Load FC matrix: (num_rois, num_rois)
        fc_matrix = np.load(fc_file)

        # Convert to tensor
        fc_tensor = torch.from_numpy(fc_matrix).float()

        if self.return_format == 'bnt':
            # For BNT: each row is a node feature
            # Input shape: (num_rois, num_rois) where each row represents a node's connectivity pattern
            return {
                "node_feature": fc_tensor,  # (num_rois, num_rois)
                "subject_name": subject_id,
                "target": target,
                "sex": sex,
            }
        else:
            # Standard format
            return {
                "fc_matrix": fc_tensor,
                "subject_name": subject_id,
                "target": target,
                "sex": sex,
            }


class FCGraphDataset(Dataset):
    """
    Dataset for functional connectivity as PyTorch Geometric graphs.

    Converts FC matrices to graph format for BrainGNN.
    """

    def __init__(
        self,
        root,
        subject_dict,
        atlas_name='cc200',
        fc_type='partial_correlation',
        threshold=None,  # Threshold for edge pruning (None = keep all)
        use_augmentations=False,
        train=True,
        **kwargs
    ):
        """
        Args:
            root: Root directory containing FC data
            subject_dict: Dictionary mapping subject IDs to (sex, target) tuples
            atlas_name: Name of brain atlas used
            fc_type: Type of functional connectivity
            threshold: Threshold for edge pruning (keep edges with |weight| > threshold)
            use_augmentations: Whether to apply data augmentation
            train: Whether this is training set
        """
        super().__init__()
        self.root = root
        self.subject_dict = subject_dict
        self.atlas_name = atlas_name
        self.fc_type = fc_type
        self.threshold = threshold
        self.use_augmentations = use_augmentations
        self.train = train

        self.data = self._set_data()

    def _set_data(self):
        """Build list of (subject_id, fc_path, corr_path, target, sex) tuples."""
        data = []
        fc_root = os.path.join(self.root, 'fc', self.atlas_name, self.fc_type)
        corr_root = os.path.join(self.root, 'fc', self.atlas_name, 'correlation')

        for subject_id in self.subject_dict:
            sex, target = self.subject_dict[subject_id]

            # Look for FC files (partial correlation for edges, correlation for node features)
            fc_file = os.path.join(fc_root, f"{subject_id}_{self.atlas_name}_{self.fc_type}.npy")
            corr_file = os.path.join(corr_root, f"{subject_id}_{self.atlas_name}_correlation.npy")

            if os.path.exists(fc_file) and os.path.exists(corr_file):
                data.append((subject_id, fc_file, corr_file, target, sex))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subject_id, fc_file, corr_file, target, sex = self.data[idx]

        # Load partial correlation for edges
        pcorr = np.load(fc_file)
        pcorr = np.abs(pcorr)  # Use absolute values

        # Load correlation for node features
        corr = np.load(corr_file)

        num_nodes = pcorr.shape[0]

        # Convert to graph
        G = from_numpy_array(pcorr)
        A = nx.to_scipy_sparse_matrix(G)
        adj = A.tocoo()

        # Extract edges and edge attributes
        edge_att = np.zeros(len(adj.row))
        for i in range(len(adj.row)):
            edge_att[i] = pcorr[adj.row[i], adj.col[i]]

        edge_index = np.stack([adj.row, adj.col])
        edge_index, edge_att = remove_self_loops(
            torch.from_numpy(edge_index).long(),
            torch.from_numpy(edge_att).float()
        )

        # Coalesce edges
        edge_index, edge_att = coalesce(
            edge_index, edge_att, num_nodes, num_nodes
        )

        # Apply threshold if specified
        if self.threshold is not None:
            mask = edge_att.abs() > self.threshold
            edge_index = edge_index[:, mask]
            edge_att = edge_att[mask]

        # Node features (correlation matrix)
        x = torch.from_numpy(corr).float()

        # Position encoding (identity matrix for ROI positions)
        pos = torch.eye(num_nodes).float()

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_att.unsqueeze(-1),
            y=torch.tensor([target]).long(),
            pos=pos
        )

        # Add metadata
        data.subject_name = subject_id
        data.sex = sex

        return data


# Dataset class mapping for easy access
DATASET_CLASSES = {
    'roi': ROIDataset,
    'fc': FCDataset,
    'fc_graph': FCGraphDataset,
}


def get_dataset_class(data_type):
    """
    Get dataset class by data type.

    Args:
        data_type: 'roi', 'fc', or 'fc_graph'

    Returns:
        Dataset class
    """
    if data_type not in DATASET_CLASSES:
        raise ValueError(f"Unknown data type: {data_type}. Choose from {list(DATASET_CLASSES.keys())}")
    return DATASET_CLASSES[data_type]
