"""
Hardware Driver for Inverted Pendulum - Raspberry Pi Zero W
Optimized for limited resources
"""

import time
import threading
import numpy as np

try:
    import pigpio
    PIGPIO_AVAILABLE = True
except ImportError:
    print("Warning: pigpio not available. Install with: sudo pip3 install pigpio")
    PIGPIO_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO not available. Using mock mode.")
    GPIO_AVAILABLE = False


class A4988StepperDriver:
    """
    A4988 Stepper Motor Driver using hardware PWM on GPIO 18
    Optimized for Pi Zero W
    """
    
    def __init__(self, step_pin=18, dir_pin=23, enable_pin=22,
                 steps_per_rev=200, microsteps=16):
        """
        Args:
            step_pin: GPIO 18 (hardware PWM) for STEP signal
            dir_pin: GPIO 23 for direction
            enable_pin: GPIO 22 for ENABLE (active LOW)
            steps_per_rev: 200 for NEMA 17
            microsteps: 16 recommended
        """
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.enable_pin = enable_pin
        self.steps_per_rev = steps_per_rev * microsteps
        
        # Current state
        self.current_position = 0
        self.current_velocity = 0.0
        self.target_velocity = 0.0
        
        # Performance tuning for Pi Zero W
        self.max_velocity = 800  # steps/sec (conservative for Pi Zero W)
        self.velocity_scale = 4.0  # force to velocity conversion
        
        # Initialize pigpio for hardware PWM
        if PIGPIO_AVAILABLE:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise RuntimeError("pigpio daemon not running. Start with: sudo pigpiod")
            
            self.pi.set_mode(self.step_pin, pigpio.OUTPUT)
            self.pi.set_mode(self.dir_pin, pigpio.OUTPUT)
            self.pi.set_mode(self.enable_pin, pigpio.OUTPUT)
            
            # Disable motor initially
            self.pi.write(self.enable_pin, 1)
        else:
            self.pi = None
            print("Motor driver in simulation mode (no pigpio)")
    
    def enable(self):
        """Enable motor (ENABLE pin LOW)"""
        if self.pi:
            self.pi.write(self.enable_pin, 0)
    
    def disable(self):
        """Disable motor (ENABLE pin HIGH)"""
        if self.pi:
            self.pi.write(self.enable_pin, 1)
            self.pi.hardware_PWM(self.step_pin, 0, 0)
    
    def set_velocity(self, velocity_steps_per_sec):
        """Set motor velocity using hardware PWM"""
        self.target_velocity = np.clip(velocity_steps_per_sec,
                                       -self.max_velocity,
                                       self.max_velocity)
        
        if self.pi:
            if abs(self.target_velocity) > 1.0:
                freq = int(abs(self.target_velocity))
                direction = 1 if self.target_velocity > 0 else 0
                
                self.pi.write(self.dir_pin, direction)
                self.pi.hardware_PWM(self.step_pin, freq, 500000)  # 50% duty cycle
                
                # Update position estimate
                self.current_velocity = self.target_velocity
            else:
                self.pi.hardware_PWM(self.step_pin, 0, 0)
                self.current_velocity = 0.0
    
    def force_to_velocity(self, force):
        """Convert control force to motor velocity"""
        return force * self.velocity_scale
    
    def apply_force(self, force):
        """Apply control force"""
        velocity = self.force_to_velocity(force)
        self.set_velocity(velocity)
    
    def get_position_estimate(self):
        """Estimate position from velocity integration"""
        return self.current_position
    
    def cleanup(self):
        """Cleanup resources"""
        if self.pi:
            self.disable()
            self.pi.stop()


class RotaryEncoder:
    """
    Quadrature encoder reader using interrupts
    Optimized for Pi Zero W
    """
    
    def __init__(self, pin_a=17, pin_b=27, pulses_per_rev=60):
        """
        Args:
            pin_a: GPIO 17 for encoder channel A
            pin_b: GPIO 27 for encoder channel B
            pulses_per_rev: 60 for your encoder
        """
        self.pin_a = pin_a
        self.pin_b = pin_b
        self.pulses_per_rev = pulses_per_rev
        
        # State
        self.position = 0
        self.last_position = 0
        self.last_time = time.time()
        self.velocity = 0.0
        
        # Velocity filtering (exponential moving average)
        self.velocity_alpha = 0.3
        
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin_a, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.pin_b, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            self.last_a = GPIO.input(self.pin_a)
            self.last_b = GPIO.input(self.pin_b)
            
            # Add interrupts
            GPIO.add_event_detect(self.pin_a, GPIO.BOTH, 
                                callback=self._callback_a)
            GPIO.add_event_detect(self.pin_b, GPIO.BOTH, 
                                callback=self._callback_b)
    
    def _callback_a(self, channel):
        """Interrupt handler for channel A"""
        if not GPIO_AVAILABLE:
            return
        
        a = GPIO.input(self.pin_a)
        b = GPIO.input(self.pin_b)
        
        if a != self.last_a:
            if a == b:
                self.position += 1
            else:
                self.position -= 1
        
        self.last_a = a
    
    def _callback_b(self, channel):
        """Interrupt handler for channel B"""
        if not GPIO_AVAILABLE:
            return
        
        a = GPIO.input(self.pin_a)
        b = GPIO.input(self.pin_b)
        
        if b != self.last_b:
            if a != b:
                self.position += 1
            else:
                self.position -= 1
        
        self.last_b = b
    
    def get_angle(self):
        """Get angle in radians"""
        angle_raw = (self.position / self.pulses_per_rev) * 2 * np.pi
        # Normalize to [-pi, pi]
        return np.arctan2(np.sin(angle_raw), np.cos(angle_raw))
    
    def get_velocity(self):
        """Get angular velocity with filtering"""
        now = time.time()
        dt = now - self.last_time
        
        if dt > 0.002:  # Update every 2ms
            delta_pos = self.position - self.last_position
            delta_angle = (delta_pos / self.pulses_per_rev) * 2 * np.pi
            instant_vel = delta_angle / dt if dt > 0 else 0.0
            
            # Exponential smoothing
            self.velocity = (self.velocity_alpha * instant_vel + 
                           (1 - self.velocity_alpha) * self.velocity)
            
            self.last_position = self.position
            self.last_time = now
        
        return self.velocity
    
    def reset(self):
        """Reset encoder to zero"""
        self.position = 0
        self.last_position = 0
        self.velocity = 0.0
    
    def cleanup(self):
        """Cleanup GPIO"""
        if GPIO_AVAILABLE:
            GPIO.remove_event_detect(self.pin_a)
            GPIO.remove_event_detect(self.pin_b)


class InvertedPendulumHardware:
    """
    Complete hardware interface for inverted pendulum on Pi Zero W
    """
    
    def __init__(self, cart_position_scale=0.01):
        """
        Args:
            cart_position_scale: meters per motor step (tune this!)
        """
        self.motor = A4988StepperDriver(
            step_pin=18,  # Hardware PWM
            dir_pin=23,
            enable_pin=22
        )
        
        self.encoder = RotaryEncoder(
            pin_a=17,
            pin_b=27,
            pulses_per_rev=60
        )
        
        self.cart_position_scale = cart_position_scale
        self.last_motor_position = 0
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.last_update_time = time.time()
        
        print("Hardware initialized")
    
    def get_state(self):
        """
        Get state vector [x, theta, x_dot, theta_dot]
        Returns numpy array compatible with your controllers
        """
        # Update cart position from motor
        now = time.time()
        dt = now - self.last_update_time
        
        if dt > 0:
            self.cart_velocity = self.motor.current_velocity * self.cart_position_scale
            self.cart_position += self.cart_velocity * dt
            self.last_update_time = now
        
        # Get pendulum state from encoder
        theta = self.encoder.get_angle()
        theta_dot = self.encoder.get_velocity()
        
        return np.array([self.cart_position, theta, 
                        self.cart_velocity, theta_dot], dtype=np.float32)
    
    def apply_force(self, force):
        """Apply control force"""
        self.motor.apply_force(force)
    
    def start(self):
        """Start hardware"""
        self.motor.enable()
        self.last_update_time = time.time()
        print("Hardware started")
    
    def stop(self):
        """Stop hardware"""
        self.motor.disable()
        print("Hardware stopped")
    
    def reset(self, calibrate=True):
        """Reset system"""
        self.motor.disable()
        time.sleep(0.1)
        
        self.motor.current_position = 0
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.encoder.reset()
        
        if calibrate:
            print("\n=== CALIBRATION ===")
            print("1. Position pendulum upright (vertical)")
            print("2. Center the cart")
            print("Press Enter when ready...")
            input()
            self.encoder.reset()
            self.cart_position = 0.0
            print("Calibration complete!\n")
    
    def cleanup(self):
        """Cleanup all resources"""
        self.stop()
        self.motor.cleanup()
        self.encoder.cleanup()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        print("Hardware cleaned up")


class MockHardware(InvertedPendulumHardware):
    """Mock hardware for testing without Pi"""
    
    def __init__(self):
        # Skip real hardware initialization
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.pendulum_angle = 0.05
        self.pendulum_velocity = 0.0
        self.last_force = 0.0
        
        # Physics params
        self.dt = 0.01
        self.mc = 1.0
        self.mp = 0.1
        self.length = 1.0
        self.g = 9.81
        
        print("Mock hardware initialized")
    
    def get_state(self):
        return np.array([self.cart_position, self.pendulum_angle,
                        self.cart_velocity, self.pendulum_velocity], 
                       dtype=np.float32)
    
    def apply_force(self, force):
        self.last_force = force
        
        # Simple physics update
        theta = self.pendulum_angle
        theta_dot = self.pendulum_velocity
        x_dot = self.cart_velocity
        
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        denom = self.mc + self.mp * sin_t**2
        
        x_ddot = (self.mp * self.length * theta_dot**2 * sin_t - 
                 self.mp * self.g * sin_t * cos_t + force) / denom
        
        theta_ddot = ((self.mc + self.mp) * self.g * sin_t - 
                     self.mp * self.length * theta_dot**2 * sin_t * cos_t - 
                     force * cos_t) / (self.length * denom)
        
        self.cart_position += x_dot * self.dt
        self.cart_velocity += x_ddot * self.dt
        self.pendulum_angle += theta_dot * self.dt
        self.pendulum_velocity += theta_ddot * self.dt
    
    def start(self):
        print("Mock hardware started")
    
    def stop(self):
        print("Mock hardware stopped")
    
    def reset(self, calibrate=False):
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.pendulum_angle = 0.05
        self.pendulum_velocity = 0.0
        print("Mock hardware reset")
    
    def cleanup(self):
        print("Mock hardware cleaned up")
