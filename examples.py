"""
Example usage of BabyGrad - demonstrating the autograd engine
"""

from babygrad import Value


def example_basic_operations():
    """Basic arithmetic operations with automatic differentiation"""
    print("=" * 50)
    print("Example 1: Basic Operations")
    print("=" * 50)
    
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    
    # Perform operations
    c = a * b + a**2
    print(f"a = {a.data}")
    print(f"b = {b.data}")
    print(f"c = a * b + a^2 = {c.data}")
    
    # Compute gradients
    c.backward()
    print(f"\nGradients:")
    print(f"dc/da = {a.grad} (expected: b + 2*a = 3 + 4 = 7)")
    print(f"dc/db = {b.grad} (expected: a = 2)")
    print()


def example_neuron():
    """Simulate a simple neuron with tanh activation"""
    print("=" * 50)
    print("Example 2: Simulating a Neuron")
    print("=" * 50)
    
    # Inputs
    x1 = Value(1.0, label='x1')
    x2 = Value(2.0, label='x2')
    
    # Weights
    w1 = Value(0.5, label='w1')
    w2 = Value(-0.3, label='w2')
    
    # Bias
    b = Value(0.1, label='b')
    
    # Neuron computation: output = tanh(w1*x1 + w2*x2 + b)
    n = w1*x1 + w2*x2 + b
    output = n.tanh()
    
    print(f"Inputs: x1={x1.data}, x2={x2.data}")
    print(f"Weights: w1={w1.data}, w2={w2.data}")
    print(f"Bias: b={b.data}")
    print(f"Output: {output.data:.4f}")
    
    # Compute gradients
    output.backward()
    
    print(f"\nGradients (for backpropagation/training):")
    print(f"∂output/∂w1 = {w1.grad:.4f}")
    print(f"∂output/∂w2 = {w2.grad:.4f}")
    print(f"∂output/∂b  = {b.grad:.4f}")
    print()


def example_complex_function():
    """Complex mathematical function"""
    print("=" * 50)
    print("Example 3: Complex Function")
    print("=" * 50)
    
    # f(x, y) = (x^2 + y)^2 / (x + 1)
    x = Value(2.0, label='x')
    y = Value(3.0, label='y')
    
    numerator = (x**2 + y)**2
    denominator = x + 1
    f = numerator / denominator
    
    print(f"f(x, y) = (x^2 + y)^2 / (x + 1)")
    print(f"x = {x.data}, y = {y.data}")
    print(f"f({x.data}, {y.data}) = {f.data:.4f}")
    
    # Compute gradients
    f.backward()
    
    print(f"\nGradients:")
    print(f"∂f/∂x = {x.grad:.4f}")
    print(f"∂f/∂y = {y.grad:.4f}")
    print()


def example_activation_functions():
    """Demonstrate various activation functions"""
    print("=" * 50)
    print("Example 4: Activation Functions")
    print("=" * 50)
    
    x_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    print("x\t\tReLU\t\ttanh\t\tsigmoid")
    print("-" * 60)
    
    for x_val in x_values:
        x = Value(x_val)
        relu_val = x.relu().data
        tanh_val = x.tanh().data
        sigmoid_val = x.sigmoid().data
        print(f"{x_val:.1f}\t\t{relu_val:.4f}\t\t{tanh_val:.4f}\t\t{sigmoid_val:.4f}")
    print()


def example_gradient_descent():
    """Simple gradient descent optimization"""
    print("=" * 50)
    print("Example 5: Gradient Descent")
    print("=" * 50)
    
    # Minimize f(x) = (x - 3)^2
    # The minimum is at x = 3
    
    x = Value(0.0, label='x')
    learning_rate = 0.1
    
    print("Minimizing f(x) = (x - 3)^2")
    print("Starting at x = 0.0\n")
    
    for i in range(20):
        # Forward pass
        target = Value(3.0)
        loss = (x - target) ** 2
        
        # Backward pass
        loss.backward()
        
        # Update (gradient descent)
        x.data -= learning_rate * x.grad
        
        # Reset gradient for next iteration
        x.zero_grad()
        
        if i % 5 == 0:
            print(f"Iteration {i:2d}: x = {x.data:.4f}, loss = {loss.data:.4f}")
    
    print(f"\nFinal: x = {x.data:.4f} (target: 3.0)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("BabyGrad - Automatic Differentiation Examples")
    print("=" * 50 + "\n")
    
    example_basic_operations()
    example_neuron()
    example_complex_function()
    example_activation_functions()
    example_gradient_descent()
    
    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)
