# babygrad

A minimalist automatic differentiation library built from scratch in pure Python. BabyGrad implements reverse-mode automatic differentiation (backpropagation) for scalar values, making it perfect for learning how neural networks compute gradients.

## Features

- **Pure Python**: No dependencies, just Python's standard library
- **Scalar-based autograd**: Track operations on individual scalars with automatic gradient computation
- **Intuitive API**: Natural Python operators (`+`, `-`, `*`, `/`, `**`)
- **Common activations**: ReLU, tanh, sigmoid, and more
- **Educational**: Simple, readable code perfect for understanding autograd

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from babygrad import Value

# Create values
a = Value(2.0)
b = Value(3.0)

# Perform operations
c = a * b + a**2
print(f"c.data = {c.data}")  # c.data = 10.0

# Compute gradients
c.backward()
print(f"dc/da = {a.grad}")  # dc/da = 7.0 (derivative with respect to a)
print(f"dc/db = {b.grad}")  # dc/db = 2.0 (derivative with respect to b)
```

## Usage Examples

### Basic Arithmetic

```python
from babygrad import Value

x = Value(2.0)
y = Value(3.0)

# Addition
z = x + y  # 5.0

# Multiplication
z = x * y  # 6.0

# Division
z = x / y  # 0.667

# Power
z = x ** 3  # 8.0

# Mixed operations
z = (x + y) * (x - y)  # 5.0 * (-1.0) = -5.0
```

### Activation Functions

```python
from babygrad import Value

x = Value(0.5)

# Hyperbolic tangent
y = x.tanh()

# ReLU (Rectified Linear Unit)
y = x.relu()

# Sigmoid
y = x.sigmoid()

# Exponential
y = x.exp()

# Natural logarithm
y = x.log()
```

### Computing Gradients

```python
from babygrad import Value

# Define a simple function: f(x, y) = x^2 + 2xy + y^2
x = Value(3.0)
y = Value(4.0)
z = x**2 + 2*x*y + y**2

# Compute gradients
z.backward()

print(f"dz/dx = {x.grad}")  # 14.0 (2x + 2y = 2*3 + 2*4)
print(f"dz/dy = {y.grad}")  # 14.0 (2x + 2y = 2*3 + 2*4)
```

### Simulating a Neuron

```python
from babygrad import Value

# Inputs
x1 = Value(1.0)
x2 = Value(2.0)

# Weights
w1 = Value(0.5)
w2 = Value(-0.3)

# Bias
b = Value(0.1)

# Neuron computation: tanh(w1*x1 + w2*x2 + b)
n = w1*x1 + w2*x2 + b
output = n.tanh()

# Compute gradients
output.backward()

print(f"Output: {output.data}")
print(f"Gradient w.r.t w1: {w1.grad}")
print(f"Gradient w.r.t w2: {w2.grad}")
print(f"Gradient w.r.t b: {b.grad}")
```

## API Reference

### Value Class

**Constructor:**
- `Value(data, _children=(), _op='', label='')`: Create a new Value object

**Attributes:**
- `data`: The scalar value (float)
- `grad`: The gradient of this value
- `label`: Optional label for debugging

**Methods:**
- `backward()`: Compute gradients using backpropagation
- `zero_grad()`: Reset gradient to zero
- `exp()`: Exponential function (e^x)
- `log()`: Natural logarithm
- `tanh()`: Hyperbolic tangent activation
- `relu()`: ReLU activation (max(0, x))
- `sigmoid()`: Sigmoid activation (1 / (1 + e^(-x)))

**Operators:**
- `+`, `-`, `*`, `/`, `**`: Arithmetic operations
- Supports operations with both Value objects and scalars

## How It Works

BabyGrad uses reverse-mode automatic differentiation:

1. **Forward Pass**: Operations build a computation graph, where each Value node tracks its inputs and the operation that created it.

2. **Backward Pass**: Starting from the output, gradients are propagated backwards through the graph using the chain rule.

```python
# Forward pass builds the graph
x = Value(2.0)
y = x * 3 + 1  # Creates: x -> (*3) -> (+1) -> y

# Backward pass computes gradients
y.backward()  # Propagates gradient from y back to x
```

## Development

### Running Tests

```bash
pytest tests/
```

With coverage:
```bash
pytest tests/ --cov=babygrad --cov-report=html
```

### Project Structure

```
babygrad/
├── babygrad/           # Main package
│   ├── __init__.py    # Package initialization
│   └── engine.py      # Core autograd engine
├── tests/             # Test suite
│   ├── __init__.py
│   └── test_engine.py # Engine tests
├── setup.py           # Package setup
├── requirements.txt   # Dependencies
├── README.md          # This file
└── LICENSE           # MIT License
```

## Mathematical Background

BabyGrad implements the **chain rule** for automatic differentiation:

If `z = f(y)` and `y = g(x)`, then:
```
dz/dx = (dz/dy) * (dy/dx)
```

For example, if `z = (x + 1)^2`:
- Forward: x=2 → y=3 → z=9
- Backward: dz/dz=1 → dz/dy=2y=6 → dz/dx=6*1=6

## Limitations

- **Scalars only**: Currently operates on scalar values, not tensors
- **Python speed**: Pure Python implementation (not optimized for performance)
- **Educational focus**: Designed for learning, not production use

## Contributing

Contributions are welcome! This is an educational project, so clarity and simplicity are valued over performance.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy, which demonstrates the fundamentals of neural network backpropagation in a minimal, educational implementation.
