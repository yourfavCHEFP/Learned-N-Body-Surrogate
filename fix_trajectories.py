import re

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
    
    # Replace the two-line pattern with single line
    content = re.sub(
        r'times, pos, vel, masses = load_trajectory\((trajectory_path|train_path|val_path|test_path|path)\)\s+graphs, normalizer = trajectory_to_graphs\(times, pos, vel, masses\)',
        r'graphs, normalizer = trajectory_to_graphs(\1)',
        content
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath}")

print("\nAll files updated!")
