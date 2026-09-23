def train_with_rollout_loss(model, train_loader, rollout_steps=10):
    """Train with multi-step rollout loss."""

    for epoch in range(max_epochs):
        for batch in train_loader:
            # One-step loss
            loss_onestep = compute_onestep_loss(model, batch)

            # Rollout loss (every 5 batches to save compute)
            if batch_idx % 5 == 0:
                loss_rollout = compute_rollout_loss(model, batch, steps=rollout_steps)
                loss = loss_onestep + 0.1 * loss_rollout
            else:
                loss = loss_onestep

            loss.backward()
            optimizer.step()
