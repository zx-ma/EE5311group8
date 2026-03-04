"""
Neural Network Policy for Inverted Pendulum
Loads weights from nn_weights.npz
Optimized for Pi Zero W (4→256→256→1 with ReLU)
"""

import numpy as np
import time


class NeuralPolicy:
    """
    Neural network policy: 4 -> 256 -> 256 -> 1
    Activation: ReLU
    Input: [x, x_dot, theta, theta_dot] (cart-pole state)
    Output: force
    """
    
    def __init__(self, weights_path='nn_weights.npz'):
        """Load weights from npz file"""
        print(f"Loading neural network from {weights_path}...")
        
        data = np.load(weights_path)
        
        # Load weights (already in correct shape)
        self.W1 = data['W1']  # (256, 4)
        self.b1 = data['b1']  # (256,)
        self.W2 = data['W2']  # (256, 256)
        self.b2 = data['b2']  # (256,)
        self.W3 = data['W3']  # (1, 256)
        self.b3 = data['b3']  # (1,)
        
        print(f"  Network: {self.W1.shape[1]} -> {self.W1.shape[0]} -> " +
              f"{self.W2.shape[0]} -> {self.W3.shape[0]}")
        print(f"  Weights dtype: {self.W1.dtype}")
        
        # Pre-allocate buffers for inference (avoid allocations in loop)
        self.h1 = np.zeros(256, dtype=np.float32)
        self.h2 = np.zeros(256, dtype=np.float32)
        
        print("Neural network loaded successfully!")
    
    def forward(self, state):
        """
        Forward pass through network
        
        Args:
            state: [x, theta, x_dot, theta_dot] (standard cart-pole state)
        
        Returns:
            force: scalar control force
        """
        # Note: Your ESP32 code expects [x, x_dot, theta, theta_dot]
        # But standard cart-pole is [x, theta, x_dot, theta_dot]
        # Let's use the standard convention and rearrange if needed
        
        # Convert to [x, x_dot, theta, theta_dot] for compatibility with trained weights
        x_in = np.array([state[0], state[2], state[1], state[3]], dtype=np.float32)
        
        # Layer 1: h1 = ReLU(W1 @ x + b1)
        np.dot(self.W1, x_in, out=self.h1)
        self.h1 += self.b1
        np.maximum(self.h1, 0, out=self.h1)  # ReLU in-place
        
        # Layer 2: h2 = ReLU(W2 @ h1 + b2)
        np.dot(self.W2, self.h1, out=self.h2)
        self.h2 += self.b2
        np.maximum(self.h2, 0, out=self.h2)  # ReLU in-place
        
        # Output layer: out = W3 @ h2 + b3
        output = np.dot(self.W3, self.h2) + self.b3
        
        return float(output[0])
    
    def __call__(self, state):
        """Make policy callable"""
        return self.forward(state)
    
    def benchmark(self, n_iterations=1000):
        """Benchmark inference speed"""
        print(f"\nBenchmarking inference speed ({n_iterations} iterations)...")
        
        dummy_state = np.array([0.1, 0.05, 0.0, 0.0], dtype=np.float32)
        
        # Warm up
        for _ in range(10):
            self.forward(dummy_state)
        
        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.time()
            self.forward(dummy_state)
            times.append(time.time() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        print(f"  Mean inference time: {np.mean(times):.3f} ms")
        print(f"  Std deviation: {np.std(times):.3f} ms")
        print(f"  Min: {np.min(times):.3f} ms, Max: {np.max(times):.3f} ms")
        print(f"  Can run at: ~{1000/np.mean(times):.0f} Hz")
        
        if np.mean(times) < 5.0:
            print("  ✓ Fast enough for 100 Hz control loop!")
        elif np.mean(times) < 10.0:
            print("  ⚠ Use 50-100 Hz control loop")
        else:
            print("  ⚠ May need to use 50 Hz or slower control loop")
        
        return np.mean(times)


def test_policy():
    """Test the neural policy"""
    print("=" * 60)
    print("Testing Neural Network Policy")
    print("=" * 60)
    
    # Load policy
    policy = NeuralPolicy('nn_weights.npz')
    
    # Test with sample states
    test_states = [
        [0.0, 0.0, 0.0, 0.0],      # Balanced at center
        [0.1, 0.05, 0.0, 0.0],     # Slightly off-balance
        [0.0, 0.1, 0.0, 0.0],      # Tilted
        [-0.1, -0.05, 0.1, 0.0],   # Cart left, tilted
    ]
    
    print("\nTesting sample states:")
    for i, state in enumerate(test_states):
        force = policy(state)
        print(f"  State {i+1}: x={state[0]:+.2f}, θ={state[1]:+.2f}, " +
              f"ẋ={state[2]:+.2f}, θ̇={state[3]:+.2f} → Force: {force:+.3f}")
    
    # Benchmark
    policy.benchmark()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_policy()
