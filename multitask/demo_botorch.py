import torch
from botorch.models import MultiTaskGP
from botorch.models.transforms import Standardize


X1, X2 = torch.rand(10, 2), torch.rand(20, 2)
t1, t2 = torch.zeros(10, 1), torch.ones(20, 1)
train_X = torch.cat([
    torch.cat([X1, t1], -1), torch.cat([X2, t2], -1),
])
print()
# train_Y = torch.cat([f1(X1), f2(X2)]).unsqueeze(-1)
# model = MultiTaskGP(train_X, train_Y, task_feature=-1)