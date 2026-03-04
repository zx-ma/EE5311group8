"""
Unified Controller Runner for Inverted Pendulum
Supports: Neural Network, PID, and Manual testing
Optimized for Raspberry Pi Zero W
"""

import time
import numpy as np
import argparse
import sys

from hardware_driver import InvertedPendulumHardware, MockHardware

# Try to import NN policy
try:
    from nn_policy import NeuralPolicy
    NN_AVAILABLE = True
except Exception as e:
    print(f"Neural network not available: {e}")
    NN_AVAILABLE = False

# Import swing-up controller
try:
    from swing_up_controller import SwingUpController, run_swing_up_then_balance
    SWING_UP_AVAILABLE = True
except Exception as e:
    print(f"Swing-up controller not available: {e}")
    SWING_UP_AVAILABLE = False


class PIDController:
    """PID Controller for inverted pendulum"""
    
    def __init__(self, Kp_theta=50.0, Kd_theta=10.0, Kp_x=5.0, Kd_x=5.0):
        """
        Args:
            Kp_theta: Proportional gain for angle
            Kd_theta: Derivative gain for angle
            Kp_x: Proportional gain for position
            Kd_x: Derivative gain for position
        """
        self.params = np.array([Kp_theta, Kd_theta, Kp_x, Kd_x])
        self.error_integral = 0.0
        
        print(f"PID Controller initialized:")
        print(f"  Kp_theta={Kp_theta}, Kd_theta={Kd_theta}")
        print(f"  Kp_x={Kp_x}, Kd_x={Kd_x}")
    
    def compute(self, state, dt=0.01):
        """
        Compute control force
        
        Args:
            state: [x, theta, x_dot, theta_dot]
            dt: timestep
        
        Returns:
            force: control force
        """
        x, theta, x_dot, theta_dot = state
        Kp_theta, Kd_theta, Kp_x, Kd_x = self.params
        
        # PID control law
        # Positive force pushes cart right, tilts pole left (reduces positive theta)
        force = (Kp_theta * theta + 
                Kd_theta * theta_dot + 
                Kp_x * (0 - x) + 
                Kd_x * (0 - x_dot))
        
        return np.clip(force, -200.0, 200.0)
    
    def __call__(self, state):
        """Make controller callable"""
        return self.compute(state)


class ManualController:
    """Manual control for testing"""
    
    def __init__(self):
        self.force = 0.0
        print("Manual Controller - Use keyboard to control")
        print("  Commands: +/- to increase/decrease force, 0 to stop, q to quit")
    
    def __call__(self, state):
        return self.force
    
    def set_force(self, force):
        self.force = np.clip(force, -200.0, 200.0)


def run_controller(hardware, controller, controller_name, 
                   duration=10.0, dt=0.01, verbose=True, log_data=True):
    """
    Run a controller on hardware
    
    Args:
        hardware: Hardware interface
        controller: Controller object (must be callable)
        controller_name: Name for display/logging
        duration: How long to run (seconds)
        dt: Control loop timestep
        verbose: Print status
        log_data: Save data to file
    
    Returns:
        states, forces, success
    """
    hardware.start()
    
    states = []
    forces = []
    times = []
    loop_times = []
    
    start_time = time.time()
    iteration = 0
    
    print(f"\n{'='*60}")
    print(f"Running {controller_name} controller for {duration:.1f}s")
    print(f"Control loop: {1/dt:.0f} Hz")
    print(f"{'='*60}\n")
    
    success = True
    
    try:
        while (time.time() - start_time) < duration:
            loop_start = time.time()
            
            # Get current state
            state = hardware.get_state()
            
            # Compute control
            force = float(controller(state))
            
            # Apply control
            hardware.apply_force(force)
            
            # Store data
            states.append(state.copy())
            forces.append(force)
            times.append(time.time() - start_time)
            
            # Check safety limits
            if abs(state[0]) > 1.0:  # Cart position
                print(f"\n⚠ SAFETY: Cart position exceeded limit ({state[0]:.3f}m)")
                success = False
                break
            
            if abs(state[1]) > np.pi/3:  # Pendulum angle (60 degrees)
                print(f"\n⚠ SAFETY: Pendulum angle exceeded limit ({np.degrees(state[1]):.1f}°)")
                success = False
                break
            
            # Print status
            if verbose and iteration % 10 == 0:
                x, theta, x_dot, theta_dot = state
                print(f"t={time.time()-start_time:6.2f}s | " +
                      f"x={x:+6.3f}m | θ={theta:+6.3f}rad ({np.degrees(theta):+6.1f}°) | " +
                      f"F={force:+7.2f}N")
            
            iteration += 1
            
            # Maintain control loop timing
            elapsed = time.time() - loop_start
            loop_times.append(elapsed * 1000)  # ms
            
            if elapsed < dt:
                time.sleep(dt - elapsed)
            elif iteration % 50 == 0:
                print(f"⚠ Warning: Control loop taking {elapsed*1000:.1f}ms (target {dt*1000:.1f}ms)")
    
    except KeyboardInterrupt:
        print("\n⚠ Controller interrupted by user")
        success = False
    
    finally:
        hardware.stop()
    
    # Convert to arrays
    states = np.array(states)
    forces = np.array(forces)
    times = np.array(times)
    loop_times = np.array(loop_times)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Duration: {times[-1]:.2f}s ({len(states)} steps)")
    print(f"  Control loop timing:")
    print(f"    Mean: {np.mean(loop_times):.2f}ms, Std: {np.std(loop_times):.2f}ms")
    print(f"    Max: {np.max(loop_times):.2f}ms")
    print(f"  Performance:")
    print(f"    Max |x|: {np.max(np.abs(states[:, 0])):.3f}m")
    print(f"    Max |θ|: {np.degrees(np.max(np.abs(states[:, 1]))):.1f}°")
    print(f"    Mean |F|: {np.mean(np.abs(forces)):.2f}N")
    print(f"  Status: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"{'='*60}\n")
    
    # Save data
    if log_data:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{controller_name.lower().replace(' ', '_')}_{timestamp}.npz"
        np.savez(filename,
                states=states,
                forces=forces,
                times=times,
                loop_times=loop_times,
                success=success,
                controller=controller_name)
        print(f"Data saved to {filename}\n")
    
    return states, forces, success


def test_mode(hardware):
    """Interactive test mode"""
    print("\n" + "="*60)
    print("TEST MODE - Manual Control")
    print("="*60)
    print("\nCommands:")
    print("  +/=  : Increase force by 10N")
    print("  -/_  : Decrease force by 10N")
    print("  0    : Stop (force = 0)")
    print("  s    : Show current state")
    print("  q    : Quit")
    print("\nPress Enter after each command\n")
    
    hardware.start()
    manual = ManualController()
    
    try:
        while True:
            state = hardware.get_state()
            force = manual.force
            hardware.apply_force(force)
            
            cmd = input(f"Force={force:+.1f}N >> ").strip().lower()
            
            if cmd in ['+', '=']:
                manual.set_force(force + 10.0)
            elif cmd in ['-', '_']:
                manual.set_force(force - 10.0)
            elif cmd == '0':
                manual.set_force(0.0)
            elif cmd == 's':
                x, theta, x_dot, theta_dot = state
                print(f"\nCurrent State:")
                print(f"  Cart position: {x:+.3f}m")
                print(f"  Pendulum angle: {theta:+.3f}rad ({np.degrees(theta):+.1f}°)")
                print(f"  Cart velocity: {x_dot:+.3f}m/s")
                print(f"  Angular velocity: {theta_dot:+.3f}rad/s\n")
            elif cmd == 'q':
                break
            else:
                print("Unknown command")
    
    except KeyboardInterrupt:
        print("\nTest mode interrupted")
    
    finally:
        hardware.stop()


def main():
    parser = argparse.ArgumentParser(description='Inverted Pendulum Controller')
    parser.add_argument('mode', choices=['nn', 'pid', 'test', 'benchmark', 'swing-nn', 'swing-pid'],
                       help='Control mode: nn, pid, test, benchmark, swing-nn (swing-up then NN), swing-pid (swing-up then PID)')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock hardware (no real Pi needed)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Run duration in seconds (default: 10)')
    parser.add_argument('--freq', type=float, default=100.0,
                       help='Control frequency in Hz (default: 100)')
    parser.add_argument('--no-calibrate', action='store_true',
                       help='Skip calibration step')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Initialize hardware
    print("\n" + "="*60)
    print("INVERTED PENDULUM CONTROL SYSTEM")
    print("Raspberry Pi Zero W")
    print("="*60 + "\n")
    
    if args.mock:
        print("Using MOCK hardware (simulation mode)\n")
        hardware = MockHardware()
    else:
        print("Using REAL hardware\n")
        try:
            hardware = InvertedPendulumHardware()
        except Exception as e:
            print(f"Error initializing hardware: {e}")
            print("Falling back to mock mode...")
            hardware = MockHardware()
    
    # Calibrate
    if not args.no_calibrate and not args.mock:
        hardware.reset(calibrate=True)
    else:
        hardware.reset(calibrate=False)
    
    dt = 1.0 / args.freq
    
    try:
        # Select controller
        if args.mode == 'nn':
            if not NN_AVAILABLE:
                print("Error: Neural network not available!")
                print("Make sure nn_weights.npz is in the current directory")
                return 1
            
            controller = NeuralPolicy('nn_weights.npz')
            
            if args.mode == 'benchmark':
                controller.benchmark()
                return 0
            
            states, forces, success = run_controller(
                hardware, controller, "Neural Network",
                duration=args.duration, dt=dt, 
                verbose=not args.quiet
            )
        
        elif args.mode == 'pid':
            # You can tune these parameters
            controller = PIDController(
                Kp_theta=50.0,
                Kd_theta=10.0,
                Kp_x=5.0,
                Kd_x=5.0
            )
            
            states, forces, success = run_controller(
                hardware, controller, "PID",
                duration=args.duration, dt=dt,
                verbose=not args.quiet
            )
        
        elif args.mode == 'test':
            test_mode(hardware)
            return 0
        
        elif args.mode == 'benchmark':
            if NN_AVAILABLE:
                policy = NeuralPolicy('nn_weights.npz')
                policy.benchmark()
            else:
                print("Neural network not available for benchmarking")
            return 0
        
        elif args.mode == 'swing-nn':
            # Swing-up then neural network balance
            if not NN_AVAILABLE:
                print("Error: Neural network not available!")
                return 1
            if not SWING_UP_AVAILABLE:
                print("Error: Swing-up controller not available!")
                return 1
            
            controller = NeuralPolicy('nn_weights.npz')
            
            states, forces, switched, success = run_swing_up_then_balance(
                hardware, controller, "Neural Network",
                duration=args.duration, dt=dt,
                verbose=not args.quiet
            )
        
        elif args.mode == 'swing-pid':
            # Swing-up then PID balance
            if not SWING_UP_AVAILABLE:
                print("Error: Swing-up controller not available!")
                return 1
            
            controller = PIDController(
                Kp_theta=500.0,
                Kd_theta=80.0,
                Kp_x=50.0,
                Kd_x=30.0
            )
            
            states, forces, switched, success = run_swing_up_then_balance(
                hardware, controller, "PID",
                duration=args.duration, dt=dt,
                verbose=not args.quiet
            )
        
        return 0 if success else 1
    
    finally:
        hardware.cleanup()


if __name__ == "__main__":
    sys.exit(main())
