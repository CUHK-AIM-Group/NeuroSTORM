import torch.nn as nn

#################################################################################################
# Contrastice loss code adapted from TCLR: Temporal Contrastive Learning for Video Representation
# https://github.com/DAVEISHAN/TCLR
#################################################################################################


class emb_head(nn.Module):
    def __init__(self, final_embedding_size=128, num_tokens = 196, use_normalization=True, n_local_frames=4):
        super(emb_head, self).__init__()
        self.final_embedding_size = final_embedding_size
        self.use_normalization = use_normalization
        self.fc1 = nn.Linear(num_tokens, self.final_embedding_size, bias=False)
        self.bn1 = nn.BatchNorm1d(self.final_embedding_size)
        self.temp_avg = nn.AdaptiveAvgPool1d(1)  #
        self.n_local_frames = n_local_frames

    def forward(self, x, type):
        # x -> b, 96, 4, 4, 4, t
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        if type == 'l':
            # Global Dense Representation, this will be used for IC and LL losses
            x = self.temp_avg(x.transpose(1, 2))
            x = x.flatten(1)
            x = nn.functional.normalize(self.bn1(self.fc1(x)), p=2, dim=1)
            return x
        elif type == 'g':
            # Global Sparse Representation, this will be used for the IC loss
            gsr = self.temp_avg(x.transpose(1, 2))
            gsr = gsr.flatten(1)
            gsr = nn.functional.normalize(self.bn1(self.fc1(gsr)), p=2, dim=1)
            return gsr
        else:
            return None, None, None, None, None