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
    
    # Add missing imports if not present
    if 'compute_gravitational_accelerations' not in content:
        content = content.replace(
            'from nbody_surrogate.dataset import (',
            'from nbody_surrogate.dataset import (\n    compute_gravitational_accelerations,\n    Normalizer,'
        )
    
    # Replace the broken pattern with correct one
    # Pattern: trajectory = load_trajectory(...)\n    graphs, normalizer = trajectory_to_graphs(trajectory)
    content = re.sub(
        r'trajectory = load_trajectory\(([^)]+)\)\s*\n\s*graphs, normalizer = trajectory_to_graphs\(trajectory\)',
        r'''trajectory = load_trajectory(\1)
    accelerations = compute_gravitational_accelerations(trajectory["positions"], trajectory["masses"])
    normalizer = Normalizer.fit(masses=trajectory["masses"], positions=trajectory["positions"], velocities=trajectory["velocities"], accelerations=accelerations)
    graphs = trajectory_to_graphs(trajectory)''',
        content
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✅ Fixed {filepath}")

print("\n🎉 All files updated!")
