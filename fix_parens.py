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
    
    # Fix missing closing parenthesis: load_trajectory("path" -> load_trajectory("path")
    content = re.sub(
        r'load_trajectory\("([^"]+)"\s*$',
        r'load_trajectory("\1")',
        content,
        flags=re.MULTILINE
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath}")

print("\nAll files updated!")
