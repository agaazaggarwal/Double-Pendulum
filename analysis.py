
import numpy as np
import matplotlib.pyplot as plt

def plot_angles_time(t_exp, th1_exp, th2_exp, t_sim, th1_sim, th2_sim):
    plt.figure(figsize=(10,5))
    plt.plot(t_exp, np.degrees(th1_exp), 'r-', label='θ₁ experiment')
    plt.plot(t_exp, np.degrees(th2_exp), 'b-', label='θ₂ experiment')
    plt.plot(t_sim, np.degrees(th1_sim), 'r--', label='θ₁ theory')
    plt.plot(t_sim, np.degrees(th2_sim), 'b--', label='θ₂ theory')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.title('Angle vs Time: Experiment vs Theory')
    plt.legend()
    plt.grid(True)
    plt.savefig('angles_time.png', dpi=150)
    plt.show()

def plot_phase_space(th, omega, title):
    plt.figure()
    plt.plot(np.degrees(th), omega, 'k-', linewidth=0.8)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Angular velocity (rad/s)')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'phase_space_{title.replace(" ", "_")}.png', dpi=150)
    plt.show()

def plot_trajectory(x_vals, y_vals, pivot):
    plt.figure(figsize=(8,8))
    
    plt.plot(x_vals, y_vals, 'g-', alpha=0.7, linewidth=0.8, label='Bottom mass path')
    
    plt.plot(pivot[0], pivot[1], 'ro', markersize=10, label='Pivot')
    
    plt.plot(x_vals[0], y_vals[0], 'bo', markersize=8, label='Start')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Trajectory of Bottom Mass')
    
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig('trajectory.png', dpi=150)
    plt.show()

def plot_divergence(t, diff_theta1, diff_theta2):
    plt.figure(figsize=(8,5))
    log_diff1 = np.log10(diff_theta1 + 1e-8)
    log_diff2 = np.log10(diff_theta2 + 1e-8)
    plt.plot(t, log_diff1, label='Δθ₁')
    plt.plot(t, log_diff2, label='Δθ₂')
    plt.xlabel('Time (s)')
    plt.ylabel('log₁₀(Δθ / rad)')
    plt.title('Divergence of Nearby Trajectories (Signature of Chaos)')
    plt.legend()
    plt.grid(True)
    plt.savefig('divergence.png', dpi=150)
    plt.show()

def error_analysis(t_exp, th1_exp, th2_exp, t_sim, th1_sim, th2_sim):
    from scipy.interpolate import interp1d
    interp_th1 = interp1d(t_sim, th1_sim, kind='linear', fill_value="extrapolate")
    interp_th2 = interp1d(t_sim, th2_sim, kind='linear', fill_value="extrapolate")
    th1_sim_aligned = interp_th1(t_exp)
    th2_sim_aligned = interp_th2(t_exp)
    rms1 = np.sqrt(np.mean((th1_exp - th1_sim_aligned)**2))
    rms2 = np.sqrt(np.mean((th2_exp - th2_sim_aligned)**2))
    print("\n--- Error Analysis ---")
    print(f"RMS error θ₁: {np.degrees(rms1):.2f} degrees")
    print(f"RMS error θ₂: {np.degrees(rms2):.2f} degrees")
    print("Possible errors: friction, air drag, camera distortion, lighting changes.")