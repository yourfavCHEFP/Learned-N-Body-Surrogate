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
    
    # Fix broken trajectory_path = \n    "path"\n)
    content = re.sub(
        r'trajectory_path = \s*\n\s*"([^"]+)"\s*\n\s*\)',
        r'trajectory_path = "\1"',
        content
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath}")

print("\nAll files updated!")
