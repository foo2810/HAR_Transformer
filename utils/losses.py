import torch

class ComplementEntroyLoss(torch.nn.Module):
    def __init__(self):
        super(ComplementEntroyLoss, self).__init__()

    #def __call__(self, y_pred, y_true):
    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        n_classes = y_pred.size(1)

        y_pred_softmax = torch.nn.functional.softmax(y_pred, dim=1)
        loss_base = torch.nn.functional.cross_entropy(y_pred, y_true)
        y_pred = y_pred_softmax

        y_pred_gt = y_pred.gather(1, y_true.reshape(batch_size, 1))
        y_pred_gt = y_pred_gt.reshape((batch_size, 1))

        y_pred_stable = (1 - y_pred_gt) + torch.tensor(1e-7, dtype=torch.float).to(y_pred.device)

        px = y_pred / y_pred_stable
        log_px = torch.log(px + torch.tensor(1e-10, dtype=torch.float).to(y_pred.device))

        mask = torch.nn.functional.one_hot(y_true, num_classes=n_classes)
        mask = mask * -1 + 1
        loss_compl = px * log_px * mask
        loss_compl = torch.sum(loss_compl) / batch_size / (n_classes - 1)

        # print('base', loss_base)
        # print('compl', loss_compl)

        return loss_base + loss_compl
