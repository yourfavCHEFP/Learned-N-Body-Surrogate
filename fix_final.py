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
    
    # Replace trajectory_path = "..." with trajectory = load_trajectory("...")
    content = content.replace(
        'trajectory_path = "',
        'trajectory = load_trajectory("'
    )
    
    # Replace trajectory_to_graphs(trajectory_path) with trajectory_to_graphs(trajectory)
    content = content.replace(
        'trajectory_to_graphs(trajectory_path)',
        'trajectory_to_graphs(trajectory)'
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath}")

print("\nAll files updated!")
