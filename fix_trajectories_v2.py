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
        lines = f.readlines()
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line has load_trajectory
        if 'load_trajectory(' in line and '=' in line:
            # Extract the variable name (trajectory_path, train_path, etc.)
            match = re.search(r'load_trajectory\(([^)]+)\)', line)
            if match:
                path_var = match.group(1)
                # Skip this line and check next line for trajectory_to_graphs
                if i + 1 < len(lines) and 'trajectory_to_graphs(' in lines[i + 1]:
                    # Replace with single line
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + f'graphs, normalizer = trajectory_to_graphs({path_var})\n')
                    i += 2  # Skip both lines
                    continue
        
        new_lines.append(line)
        i += 1
    
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    print(f"Fixed {filepath}")

print("\nAll files updated!")
