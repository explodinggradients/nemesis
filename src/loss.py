import torch
from torch import nn

class RMLoss(nn.Module):
    """
    """
    def __init__(
            self,
            reduction = None,
            beta = 0.001,
    ):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(
            self,
            logits,
            k_lens=None,
    ):
        total_loss = []
        indices = list(zip(k_lens[:-1],k_lens[1:]))
        for start,end in indices:
            combinations = torch.combinations(torch.arange(start, end, device=logits.device), 2)
            positive = logits[combinations[:,0]]
            negative = logits[combinations[:,1]]
            l2 = 0.5 * (positive**2 - negative**2)
            loss = -1 * (nn.functional.logsigmoid(positive - negative) + self.beta * l2).mean()
            total_loss.append(loss)

        total_loss = torch.stack(total_loss)
        if self.reduction == "mean":
            total_loss = total_loss.mean()

        return total_loss


        
        