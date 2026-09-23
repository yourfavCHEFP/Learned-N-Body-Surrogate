#!/bin/bash
set -e

echo "============================================================"
echo "🪐 RUNNING COMPLETE LEARNED N-BODY SURROGATE PIPELINE (8-12)"
echo "============================================================"

echo ""
echo "▶️ Phase 8: Long-Horizon Stability (Residual & Rollout Training)"
echo "------------------------------------------------------------"
python train_gnn_residual.py
python train_gnn_rollout.py
python evaluate_stability.py

echo ""
echo "▶️ Phase 9: Hamiltonian Neural Networks (Exact Energy Conservation)"
echo "------------------------------------------------------------"
python train_hnn.py
python evaluate_energy_conservation.py

echo ""
echo "▶️ Phase 10: Graph Attention Networks & Interpretability"
echo "------------------------------------------------------------"
python train_gnn_attention.py
python analyze_attention_patterns.py

echo ""
echo "▶️ Phase 12: Cross-System Generalization & Failure Analysis"
echo "------------------------------------------------------------"
python generate_varied_systems.py
python evaluate_generalization.py
python analyze_failure_modes.py

echo ""
echo "============================================================"
echo "🎉 ALL ADVANCED RESEARCH PHASES COMPLETED SUCCESSFULLY!"
echo "============================================================"
