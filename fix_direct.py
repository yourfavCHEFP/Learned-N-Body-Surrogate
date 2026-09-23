files = [
    'train_gnn_residual.py',
    'train_gnn_rollout.py', 
    'train_hnn.py',
    'train_gnn_attention.py',
    'evaluate_stability.py',
    'evaluate_energy_conservation.py',
    'analyze_attention_patterns.py',
    'evaluate_generalization.py'
]

for filepath in files:
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Simply replace the 4-argument call with 1-argument call
    content = content.replace(
        'trajectory_to_graphs(times, pos, vel, masses)',
        'trajectory_to_graphs(trajectory_path)'
    )
    
    # Also need to replace load_trajectory assignment
    content = content.replace(
        'times, pos, vel, masses = load_trajectory(',
        'trajectory_path = '
    ).replace(
        'times, pos, vel, masses = load_trajectory(',
        'trajectory_path = '
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath}")

print("\nAll files updated!")
