import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import models

#################################################################################################
# Contrastice loss code adapted from TCLR: Temporal Contrastive Learning for Video Representation
# https://github.com/DAVEISHAN/TCLR
#################################################################################################

class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)
        # print(f'similarity_matrix shpae is {similarity_matrix.shape}')

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        # print(f'l_pos shpae is {l_pos.shape}')

        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


def global_local_temporal_contrastive(lsr,gdr, temperature):
    #lsr denotes local sparse-clip representation= representation of temporal slice of global clip
    #gdr denotes global dense-clip representation= representation of global(pooled) feature of local clip

    #lsr,gdr shape should be  [BS,num_clips,128]
    num_clips = lsr.shape[1]
    similarity_matrix = torch.bmm(lsr, gdr.permute(0,2,1)) # [BS, num_clips, num_clips]
    # print(similarity_matrix)
    similarity_matrix = torch.cat((similarity_matrix, similarity_matrix.permute(0,2,1)),dim=0) # [BS*2, num_clips, num_clips]
    # print()
    # print(similarity_matrix)
    similarity_matrix = similarity_matrix.view(-1, num_clips) # [BS*2*num_clips, num_clips]
    # print()
    # print(similarity_matrix)
    # print()
    sample_lab = [i for i in range(num_clips)]
    label = []
    for i in range(lsr.shape[0]*2):
        label.extend(sample_lab)
    label = torch.from_numpy(np.asarray(label)).long().cuda()
    similarity_matrix /= temperature

    loss = nn.functional.cross_entropy(similarity_matrix, label, reduction='sum')
    return loss/ (2*lsr.shape[0])


def rotation_loss(output_rot, target_rot):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output_rot, target_rot.type(torch.LongTensor).cuda()).cuda()
    return loss
