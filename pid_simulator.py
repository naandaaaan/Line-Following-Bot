
import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0, 1.6, 240)
y1 = 0.13 * np.sin(2 * np.pi * x1 / 1.6)

# Straight section before the 90-degree turn
x2 = np.linspace(1.6, 1.95, 50)
y2 = np.zeros_like(x2)

# 90-degree curved turn
angle = np.linspace(-np.pi / 2, 0, 80)
radius = 0.35

x3 = 1.95 + radius * np.cos(angle)
y3 = 0.35 + radius * np.sin(angle)

# Vertical section after the turn
x4 = np.full(100, 2.3)
y4 = np.linspace(0.35, 1.1, 100)

track_x = np.concatenate([x1, x2, x3, x4])
track_y = np.concatenate([y1, y2, y3, y4])


dt = 0.02
wheelbase = 0.12
base_speed = 0.28
max_speed = 0.45

sensor_positions = np.array([
    -0.036, -0.018, 0, 0.018, 0.036
])

sensor_offset = 0.055
sensor_sigma = 0.018



def simulate(Kp, Ki, Kd):

    # Initial robot position and orientation
    x = 0.02
    y = 0.0
    theta = 0.0

    integral = 0.0
    previous_error = 0.0
    last_error = 0.0

    history_x = []
    history_y = []
    history_error = []

    for step in range(1000):

        # Find sensor readings
        readings = []

        for pos in sensor_positions:

            # Position of each sensor in world coordinates
            sx = (x + sensor_offset * np.cos(theta)
                  - pos * np.sin(theta))

            sy = (y + sensor_offset * np.sin(theta)
                  + pos * np.cos(theta))

            # Distance from sensor to nearest track point
            distances = np.sqrt(
                (track_x - sx)**2 + (track_y - sy)**2
            )

            min_distance = np.min(distances)

            # Simulated IR sensor response
            strength = np.exp(
                -(min_distance**2) /
                (2 * sensor_sigma**2)
            )

            readings.append(strength)

        readings = np.array(readings)

        # Weighted-average position
        if np.sum(readings) > 1e-6:
            error = np.sum(
                sensor_positions * readings
            ) / np.sum(readings)
            last_error = error
        else:
            error = last_error

        # PID calculations
        integral += error * dt

        derivative = (error - previous_error) / dt

        correction = (
            Kp * error
            + Ki * integral
            + Kd * derivative
        )

        correction = np.clip(correction, -5, 5)

        # Differential drive wheel velocities
        left_speed = (
            base_speed - correction * wheelbase / 2
        )

        right_speed = (
            base_speed + correction * wheelbase / 2
        )

        left_speed = np.clip(
            left_speed, -max_speed, max_speed
        )

        right_speed = np.clip(
            right_speed, -max_speed, max_speed
        )

        # Robot linear and angular velocities
        v = (right_speed + left_speed) / 2

        omega = (
            right_speed - left_speed
        ) / wheelbase

        # Update robot position
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt

        theta += omega * dt

        # Save data for plotting
        history_x.append(x)
        history_y.append(y)
        history_error.append(error)

        previous_error = error

    return (
        history_x,
        history_y,
        history_error
    )


# ---------------------------------
# 4. RUN P-ONLY AND PID
# ---------------------------------

# P-only controller
p_result = simulate(
    Kp=22, Ki=0, Kd=0
)

# Tuned PID controller
pid_result = simulate(
    Kp=22, Ki=0.1, Kd=0.65
)


plt.figure(figsize=(12, 5))

# Trajectory comparison
plt.subplot(1, 2, 1)

plt.plot(
    track_x, track_y,
    'k--', label="Track"
)

plt.plot(
    p_result[0], p_result[1],
    label="P-only"
)

plt.plot(
    pid_result[0], pid_result[1],
    label="PID"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Robot Path Comparison")
plt.axis("equal")
plt.legend()
plt.grid()

# Error comparison
plt.subplot(1, 2, 2)

time = np.arange(1000) * dt

plt.plot(
    time, p_result[2],
    label="P-only"
)

plt.plot(
    time, pid_result[2],
    label="PID"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Position Error (m)")
plt.title("Tracking Error Comparison")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
