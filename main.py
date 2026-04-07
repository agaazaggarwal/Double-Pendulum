import numpy as np
from tracker import DoublePendulumTracker
from simulation import simulate_double_pendulum
from analysis import plot_angles_time, plot_phase_space, plot_trajectory, plot_divergence, error_analysis
from config import VIDEO_PATH

def main():
    print("=== Double Pendulum Analysis ===")
    print("Step 1: Tracking markers in video...")
    tracker = DoublePendulumTracker(VIDEO_PATH)
    t_exp, th1_exp, th2_exp, w1_exp, w2_exp, pos_top, pos_bot = tracker.track()
    print(f"✓ Tracked {len(t_exp)} frames")

    
    if not np.isfinite(th1_exp).all() or not np.isfinite(th2_exp).all():
        print("Warning: Non-finite angles detected. Using first valid frame.")
        valid = np.isfinite(th1_exp) & np.isfinite(th2_exp)
        if not np.any(valid):
            raise RuntimeError("No valid angles found.")
        th1_exp = th1_exp[valid]
        th2_exp = th2_exp[valid]
        w1_exp = w1_exp[valid]
        w2_exp = w2_exp[valid]
        t_exp = t_exp[valid]
        pos_top = pos_top[valid]
        pos_bot = pos_bot[valid]

    
    th1_0 = th1_exp[0]
    th2_0 = th2_exp[0]
    w1_0 = w1_exp[0]
    w2_0 = w2_exp[0]
    print(f"Initial angles: θ₁ = {np.degrees(th1_0):.1f}°, θ₂ = {np.degrees(th2_0):.1f}°")
    print(f"Initial angular velocities: ω₁ = {w1_0:.2f} rad/s, ω₂ = {w2_0:.2f} rad/s")

    
    t_sim = np.linspace(0, t_exp[-1], len(t_exp))
    try:
        t_sim, th1_sim, th2_sim, w1_sim, w2_sim = simulate_double_pendulum(
            th1_0, th2_0, w1_0, w2_0, t_sim
        )
    except ValueError as e:
        print(f"Simulation error: {e}")
        print("Please check that initial angles are finite and reasonable.")
        return

    
    plot_angles_time(t_exp, th1_exp, th2_exp, t_sim, th1_sim, th2_sim)
    plot_phase_space(th1_exp, w1_exp, "Phase space - Top mass (experiment)")
    plot_phase_space(th2_exp, w2_exp, "Phase space - Bottom mass (experiment)")
    plot_trajectory(pos_bot[:,0], pos_bot[:,1], tracker.pivot)

    
    print("\nDemonstrating chaos: perturbing initial θ₂ by 1°...")
    th2_pert = th2_0 + np.radians(1.0)
    _, th1_pert, th2_pert, _, _ = simulate_double_pendulum(
        th1_0, th2_pert, w1_0, w2_0, t_sim
    )
    diff_th1 = np.abs(th1_sim - th1_pert)
    diff_th2 = np.abs(th2_sim - th2_pert)
    plot_divergence(t_sim, diff_th1, diff_th2)

    
    error_analysis(t_exp, th1_exp, th2_exp, t_sim, th1_sim, th2_sim)

    print("\n✅ All done! Check the folder for PNG images.")

if __name__ == "__main__":
    main()