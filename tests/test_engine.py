"""
Tests for the BabyGrad automatic differentiation engine.
"""

import pytest
import math
from babygrad import Value


class TestValueBasics:
    """Test basic Value creation and representation"""
    
    def test_value_creation(self):
        """Test creating a Value object"""
        v = Value(3.0)
        assert v.data == 3.0
        assert v.grad == 0.0
    
    def test_value_from_int(self):
        """Test creating a Value from an integer"""
        v = Value(5)
        assert v.data == 5.0
        assert isinstance(v.data, float)
    
    def test_value_repr(self):
        """Test string representation of Value"""
        v = Value(2.5)
        assert "Value(data=2.5)" in repr(v)


class TestArithmeticOperations:
    """Test arithmetic operations"""
    
    def test_addition(self):
        """Test addition of two Values"""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0
    
    def test_addition_with_scalar(self):
        """Test addition of Value and scalar"""
        a = Value(2.0)
        c = a + 3.0
        assert c.data == 5.0
        
        c = 3.0 + a
        assert c.data == 5.0
    
    def test_multiplication(self):
        """Test multiplication of two Values"""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0
    
    def test_multiplication_with_scalar(self):
        """Test multiplication of Value and scalar"""
        a = Value(2.0)
        c = a * 3.0
        assert c.data == 6.0
        
        c = 3.0 * a
        assert c.data == 6.0
    
    def test_subtraction(self):
        """Test subtraction of two Values"""
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0
    
    def test_subtraction_with_scalar(self):
        """Test subtraction with scalar"""
        a = Value(5.0)
        c = a - 3.0
        assert c.data == 2.0
        
        c = 5.0 - a
        assert c.data == 0.0
    
    def test_division(self):
        """Test division of two Values"""
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert c.data == 2.0
    
    def test_division_with_scalar(self):
        """Test division with scalar"""
        a = Value(6.0)
        c = a / 3.0
        assert c.data == 2.0
        
        c = 6.0 / a
        assert c.data == 1.0
    
    def test_negation(self):
        """Test negation of a Value"""
        a = Value(3.0)
        b = -a
        assert b.data == -3.0
    
    def test_power(self):
        """Test power operation"""
        a = Value(2.0)
        c = a ** 3
        assert c.data == 8.0
    
    def test_power_fractional(self):
        """Test power with fractional exponent"""
        a = Value(4.0)
        c = a ** 0.5
        assert c.data == 2.0


class TestActivationFunctions:
    """Test activation functions"""
    
    def test_exp(self):
        """Test exponential function"""
        a = Value(1.0)
        b = a.exp()
        assert abs(b.data - math.e) < 1e-7
    
    def test_log(self):
        """Test natural logarithm"""
        a = Value(math.e)
        b = a.log()
        assert abs(b.data - 1.0) < 1e-7
    
    def test_tanh(self):
        """Test hyperbolic tangent"""
        a = Value(0.0)
        b = a.tanh()
        assert abs(b.data - 0.0) < 1e-7
        
        a = Value(1.0)
        b = a.tanh()
        expected = (math.exp(2) - 1) / (math.exp(2) + 1)
        assert abs(b.data - expected) < 1e-7
    
    def test_relu(self):
        """Test ReLU activation"""
        a = Value(-2.0)
        b = a.relu()
        assert b.data == 0.0
        
        a = Value(2.0)
        b = a.relu()
        assert b.data == 2.0
        
        a = Value(0.0)
        b = a.relu()
        assert b.data == 0.0
    
    def test_sigmoid(self):
        """Test sigmoid activation"""
        a = Value(0.0)
        b = a.sigmoid()
        assert abs(b.data - 0.5) < 1e-7
        
        a = Value(100.0)  # Should be close to 1
        b = a.sigmoid()
        assert b.data > 0.99


class TestGradients:
    """Test gradient computation"""
    
    def test_simple_gradient(self):
        """Test gradient of a simple operation"""
        a = Value(2.0)
        b = a * 3.0
        b.backward()
        assert a.grad == 3.0
    
    def test_addition_gradient(self):
        """Test gradient through addition"""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0
    
    def test_multiplication_gradient(self):
        """Test gradient through multiplication"""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        assert a.grad == 3.0
        assert b.grad == 2.0
    
    def test_power_gradient(self):
        """Test gradient through power operation"""
        a = Value(2.0)
        b = a ** 3
        b.backward()
        # d/dx(x^3) = 3x^2, at x=2: 3*4 = 12
        assert a.grad == 12.0
    
    def test_complex_expression(self):
        """Test gradient of a complex expression"""
        # f(x, y) = x^2 + 2*x*y + y^2
        x = Value(3.0)
        y = Value(4.0)
        z = x**2 + 2*x*y + y**2
        z.backward()
        # df/dx = 2x + 2y = 2*3 + 2*4 = 14
        # df/dy = 2x + 2y = 2*3 + 2*4 = 14
        assert x.grad == 14.0
        assert y.grad == 14.0
    
    def test_exp_gradient(self):
        """Test gradient through exponential"""
        a = Value(1.0)
        b = a.exp()
        b.backward()
        # d/dx(e^x) = e^x, at x=1: e
        assert abs(a.grad - math.e) < 1e-7
    
    def test_log_gradient(self):
        """Test gradient through logarithm"""
        a = Value(2.0)
        b = a.log()
        b.backward()
        # d/dx(ln(x)) = 1/x, at x=2: 0.5
        assert abs(a.grad - 0.5) < 1e-7
    
    def test_tanh_gradient(self):
        """Test gradient through tanh"""
        a = Value(0.5)
        b = a.tanh()
        b.backward()
        # d/dx(tanh(x)) = 1 - tanh(x)^2
        expected_grad = 1 - b.data**2
        assert abs(a.grad - expected_grad) < 1e-7
    
    def test_relu_gradient(self):
        """Test gradient through ReLU"""
        # Positive input
        a = Value(2.0)
        b = a.relu()
        b.backward()
        assert a.grad == 1.0
        
        # Negative input
        a = Value(-2.0)
        b = a.relu()
        b.backward()
        assert a.grad == 0.0
    
    def test_sigmoid_gradient(self):
        """Test gradient through sigmoid"""
        a = Value(0.0)
        b = a.sigmoid()
        b.backward()
        # d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
        # at x=0: 0.5 * 0.5 = 0.25
        assert abs(a.grad - 0.25) < 1e-7
    
    def test_zero_grad(self):
        """Test gradient reset"""
        a = Value(2.0)
        b = a * 3.0
        b.backward()
        assert a.grad == 3.0
        
        a.zero_grad()
        assert a.grad == 0.0
    
    def test_chain_rule(self):
        """Test chain rule with nested operations"""
        # f(x) = (x + 1)^2
        x = Value(2.0)
        y = x + 1
        z = y ** 2
        z.backward()
        # df/dx = 2(x + 1) = 2*3 = 6
        assert x.grad == 6.0


class TestComplexComputations:
    """Test more complex computations"""
    
    def test_neuron_like_computation(self):
        """Test a computation similar to a neuron"""
        # Simulate: y = tanh(w1*x1 + w2*x2 + b)
        x1 = Value(1.0)
        x2 = Value(2.0)
        w1 = Value(0.5)
        w2 = Value(-0.3)
        b = Value(0.1)
        
        n = w1*x1 + w2*x2 + b
        y = n.tanh()
        y.backward()
        
        # All gradients should be computed
        assert x1.grad != 0.0
        assert x2.grad != 0.0
        assert w1.grad != 0.0
        assert w2.grad != 0.0
        assert b.grad != 0.0
    
    def test_multi_layer_computation(self):
        """Test multi-layer computation"""
        # Layer 1
        x = Value(1.0)
        w1 = Value(2.0)
        h1 = (x * w1).tanh()
        
        # Layer 2
        w2 = Value(3.0)
        out = h1 * w2
        
        out.backward()
        
        # Check that gradients propagate through both layers
        assert x.grad != 0.0
        assert w1.grad != 0.0
        assert w2.grad != 0.0


class TestEdgeCases:
    """Test edge cases"""
    
    def test_zero_value(self):
        """Test operations with zero"""
        a = Value(0.0)
        b = Value(5.0)
        c = a + b
        assert c.data == 5.0
        
        c = a * b
        assert c.data == 0.0
    
    def test_negative_values(self):
        """Test operations with negative values"""
        a = Value(-2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 1.0
        
        c = a * b
        assert c.data == -6.0
    
    def test_division_by_small_number(self):
        """Test division doesn't cause overflow"""
        a = Value(1.0)
        b = Value(0.001)
        c = a / b
        assert c.data == 1000.0
