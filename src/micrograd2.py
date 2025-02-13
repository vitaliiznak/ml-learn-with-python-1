import math
import random
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._prev = _children
        self._op = _op
        self.label = label

        def _backward():
            pass
        self._backward = _backward

    @classmethod
    def topological_sort(cls, node):
        topo = []
        visited = set()

        def topological_sort_aux(node):
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    topological_sort_aux(prev)
                topo.append(node)
        topological_sort_aux(node)
        return topo

    def __repr__(self):
        return f'Value(data={self.data}, _op={self._op}, label={self.label})'

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = Value.topological_sort(self)
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f'Neuron(w={self.w}, b={self.b})'


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs if len(outs) > 1 else outs[0]

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


# Example usage
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
yt = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])

# Training loop
for k in range(10):
    ypred = [n(x) for x in xs]
    loss = sum([(ypred_el - yt_el)**2 for yt_el, ypred_el in zip(yt, ypred)])

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    for p in n.parameters():
        p.data -= 0.01 * p.grad

    print(k, 'loss:', loss.data)

# Visualization


def trace(root):
    nodes, edges = set(), set()

    def visit(node):
        if node not in nodes:
            nodes.add(node)
            for prev in node._prev:
                edges.add((prev, node))
                visit(prev)
    visit(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(uid, label=f"{n.label} | {n._op} | data: {
                 n.data:.4f} | grad: {n.grad:.4f}", shape='record')
        if n._op:
            dot.node(uid + n._op, label=n._op, shape='diamond')
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot


draw_dot(loss)
