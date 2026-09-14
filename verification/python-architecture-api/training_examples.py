"""Small native CPU training examples; no user Module __init__ boilerplate."""
import json
import ferro as fr
from ferro.nn import functional as F


class GraphModel(fr.nn.Module):
    def build(self):
        self.edges = fr.graph.COO(3, 3, [0,1,2,2], [1,0,0,1])
        self.weight = fr.nn.Parameter(fr.Tensor([.2]*4, [4]))

    def forward(self, x):
        return self.edges.spmm(self.weight.tensor(), x)


class SequenceModel(fr.nn.Module):
    def build(self):
        self.wi = fr.nn.Parameter(fr.Tensor([.3], [1,1]))
        self.wh = fr.nn.Parameter(fr.Tensor([.5], [1,1]))

    def forward(self, x):
        state = fr.recurrent.RecurrentState(fr.Tensor.zeros([2,1]))
        result = fr.recurrent.unroll('rnn', x, state, self.wi.tensor(), self.wh.tensor(), lengths=[4,4], truncate=2)
        return result.state.h


class SplineModel(fr.nn.Module):
    def build(self):
        basis = fr.basis.BSplineBasis([0,0,0,0,1,1,1,1], 3)
        self.spline = fr.nn.KanLayer(basis, fr.Tensor.zeros([1,1,4]))

    def forward(self, x):
        return self.spline(x)


class ImageModel(fr.nn.Module):
    def build(self):
        self.weight = fr.nn.Parameter(fr.Tensor([.2], [1,1,1,1]))

    def forward(self, x):
        return F.conv2d(x, self.weight.tensor())


def train(model, x, target, steps, lr):
    optimizer = fr.optim.SGD(model.parameters(), lr=lr)
    first = F.mse_loss(model(x), target).item()
    for _ in range(steps):
        optimizer.zero_grad()
        F.mse_loss(model(x), target).backward()
        optimizer.step()
    return first, F.mse_loss(model(x), target).item()


def run_examples():
    sequence = fr.Tensor([-.4,.4,0,0,0,0,0,0], [4,2,1])
    spline_x = fr.Tensor([.1,.3,.5,.7,.9], [5,1])
    image = fr.Tensor([.2,.4,.6,.8], [1,1,2,2])
    return {
        'graph': train(GraphModel(), fr.Tensor([1,2,3],[3,1]), fr.Tensor([2,1,3],[3,1]), 60, .1),
        'sequence': train(SequenceModel(), sequence, fr.Tensor([-.4,.4],[2,1]), 160, .4),
        'kan': train(SplineModel(), spline_x, spline_x * spline_x, 180, .3),
        'convolution': train(ImageModel(), image, image * 2, 80, .2),
    }


if __name__ == '__main__':
    print(json.dumps(run_examples(), indent=2))
