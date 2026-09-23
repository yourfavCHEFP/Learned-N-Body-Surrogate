def add_state_noise(positions, velocities, pos_noise_std=1e-4, vel_noise_std=1e-5):
    """Add Gaussian noise to state during training."""
    pos_noisy = positions + pos_noise_std * torch.randn_like(positions)
    vel_noisy = velocities + vel_noise_std * torch.randn_like(velocities)
    return pos_noisy, vel_noisy
