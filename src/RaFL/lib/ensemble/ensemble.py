import torch
import torch.nn as nn
# TODO Max, Avg, Majority vote(hard max [1,0,0,0])
import torch.nn.functional as F

def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) / len(outputs)

class AvgEnsemble(nn.Module):
    def __init__(self, net_list):
        super(AvgEnsemble, self).__init__()
        self.estimators = nn.ModuleList(net_list)

    def forward(self, x):
        outputs = [
            estimator(x) for estimator in self.estimators
        ]
        outputs = torch.stack(outputs)
        proba = average(outputs)

        return proba


class MaxEnsemble(nn.Module):
    def __init__(self, nets: list):
        super(MaxEnsemble, self).__init__()
        self.estimators = nn.ModuleList(nets)

    def forward(self, x):

        outputs = [
            estimator(x) for estimator in self.estimators
        ]
        outputs = torch.stack(outputs)
        return F.softmax(torch.max(outputs, 0).values, dim=1)


class VoteEnsemble(nn.Module):
    def __init__(self, nets: list):
        super(VoteEnsemble, self).__init__()
        self.nets = nets

    def forward(self, x):
        res = []
        for net in self.nets:
            res.append(net(x))
        return nn.Softmax(res)


if __name__ == '__main__':
    from RaFL.networks.vgg import vgg11

    net = vgg11()
    net1 = vgg11()
    net2 = vgg11()

    netlist = [net, net2, net1]
    nets = MaxEnsemble(netlist)
    y = nets(torch.randn(128, 3, 32, 32))

    print(y)
    y = net(torch.randn(128, 3, 32, 32))
    print(y)