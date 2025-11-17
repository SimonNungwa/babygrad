"""
Core automatic differentiation engine for BabyGrad.

This module implements the Value class which tracks operations and computes gradients
using reverse-mode automatic differentiation (backpropagation).
"""

import math


class Value:
    """
    A Value wraps a scalar and tracks the operations performed on it to enable
    automatic differentiation via backpropagation.
    
    Attributes:
        data: The scalar value (float)
        grad: The gradient of this value with respect to some output
        _backward: Function to propagate gradients to inputs
        _prev: Set of Value objects that created this Value
        _op: String describing the operation that created this Value
    """
    
    def __init__(self, data, _children=(), _op='', label=''):
        """
        Initialize a Value.
        
        Args:
            data: The scalar value (will be converted to float)
            _children: Tuple of parent Value objects
            _op: String describing the operation
            label: Optional label for debugging/visualization
        """
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        """Addition: self + other"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        """Multiplication: self * other"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        """Power: self ** other (other must be int or float)"""
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __rpow__(self, other):
        """Reverse power: other ** self"""
        return Value(other) ** self
    
    def __neg__(self):
        """Negation: -self"""
        return self * -1
    
    def __sub__(self, other):
        """Subtraction: self - other"""
        return self + (-other)
    
    def __rsub__(self, other):
        """Reverse subtraction: other - self"""
        return other + (-self)
    
    def __truediv__(self, other):
        """Division: self / other"""
        return self * other**-1
    
    def __rtruediv__(self, other):
        """Reverse division: other / self"""
        return other * self**-1
    
    def __radd__(self, other):
        """Reverse addition: other + self"""
        return self + other
    
    def __rmul__(self, other):
        """Reverse multiplication: other * self"""
        return self * other
    
    def exp(self):
        """Exponential function: e^self"""
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def log(self):
        """Natural logarithm: ln(self)"""
        x = self.data
        out = Value(math.log(x), (self,), 'log')
        
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        """Hyperbolic tangent activation function"""
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        """ReLU activation function: max(0, self)"""
        out = Value(max(0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
    
    def sigmoid(self):
        """Sigmoid activation function: 1 / (1 + e^(-self))"""
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        """
        Compute gradients using reverse-mode automatic differentiation.
        
        This method performs a topological sort of the computation graph and
        calls _backward on each node in reverse topological order to accumulate
        gradients.
        """
        # Build topological order of all children in the graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient of output to 1.0
        self.grad = 1.0
        
        # Apply chain rule going backwards through the graph
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Reset the gradient to zero"""
        self.grad = 0.0


# Utility functions
def exp(x):
    """Exponential function"""
    return x.exp() if isinstance(x, Value) else Value(x).exp()


def log(x):
    """Natural logarithm"""
    return x.log() if isinstance(x, Value) else Value(x).log()


def tanh(x):
    """Hyperbolic tangent"""
    return x.tanh() if isinstance(x, Value) else Value(x).tanh()


def relu(x):
    """ReLU activation"""
    return x.relu() if isinstance(x, Value) else Value(x).relu()


def sigmoid(x):
    """Sigmoid activation"""
    return x.sigmoid() if isinstance(x, Value) else Value(x).sigmoid()
