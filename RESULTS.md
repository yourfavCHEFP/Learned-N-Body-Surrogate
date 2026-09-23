# Learned N-Body Surrogate: Complete Results

## Executive Summary

Trained 6 neural architectures to approximate gravitational N-body dynamics. Best model (Attention GNN) achieves 23× better accuracy than standard GNN. Energy conservation improved from 193% drift to <10% via Hamiltonian constraints. Honest assessment: poor generalization to unseen systems reveals need for diverse training data.

---

## Model Comparison

| Model | Params | Val MSE | Energy Drift | Stable Steps | Interpretability |
|-------|--------|---------|--------------|--------------|------------------|
| MLP Baseline | 20K | 7.07e-6 | 5395% | ~120 | ✗ |
| Standard GNN | 137K | 4.45e-6 | 193% | ~250 | ✗ |
| Residual GNN | 137K | 3.81e-6 | 48.2% | ~390 | ✗ |
| Rollout GNN | 137K | — | 22.7% | ~460 | ✗ |
| **Hamiltonian NN** | 145K | 2.18e-6 | **<10%** | ~300 | ✗ |
| **Attention GNN** | 152K | **1.90e-7** | — | ~250 | **✓** |

---

## Key Findings

### 1. Graph Structure Matters
GNN outperforms MLP by 37% (MSE 4.45e-6 vs 7.07e-6), confirming that relational inductive biases improve dynamics learning.

### 2. Training Strategy > Architecture
Residual and Rollout training (Phase 8) extend stable horizon 84% without changing model architecture. Error accumulation is the bottleneck, not model capacity.

### 3. Physics Constraints Work
Hamiltonian NN achieves <10% energy drift vs 193% for standard GNN—a 20× improvement. Hard constraints outperform soft regularization.

### 4. Attention Reveals Physics
Learned attention weights show HIGH star-planet, LOW planet-planet interactions, matching Newtonian gravity where star dominates.

### 5. Real-System & Generalization Assessment
The project now includes real exoplanet-system preparation and generalization stress testing. Models are evaluated on synthetic and realistic configurations, including multi-planet systems and out-of-distribution tests. This reveals:
- Training on a single system causes overfitting to specific masses/distances
- Model memorizes rather than learning universal gravitational law
- **Solution:** expand to diverse systems and real exoplanet parameterization (TRAPPIST-1, Kepler-11, etc.)

### 6. Real Exoplanet Pipeline
The repository includes real-system preparation tooling for NASA Exoplanet Archive and JPL Horizons, plus validation for multi-planet systems. This moves the project beyond toy synthetic benchmarks toward physically realistic astronomy data.

---

## Failure Modes & Mitigation

| Failure Mode | Cause | Mitigation |
|--------------|-------|------------|
| Long-horizon divergence | Error accumulation | Phase 8 (Rollout training) |
| Energy drift | No conservation constraint | Phase 9 (Hamiltonian NN) |
| Different N-bodies | Fixed architecture | PyG handles variable graphs |
| Extreme mass ratios | Single training system | Diverse training data |
| High eccentricity | Low-e training distribution | Data augmentation |

---

## Computational Cost

All training on CPU (MacBook Air M2):
- Phase 8: ~30 min (2 models)
- Phase 9: ~40 min (HNN with autograd)
- Phase 10: ~25 min (Attention GNN)
- Phase 12: ~15 min (generation + evaluation)
- **Total: ~2 hours**

---

## Next Steps

1. **Train on real exoplanet systems:** TRAPPIST-1, Kepler-11 data ready in `data/simulated/`
2. **Diverse training:** Generate 1000+ varied systems (mass ratios, eccentricities)
3. **Ensemble methods:** Quantify prediction uncertainty
4. **Deploy:** GitHub Pages visualization of live N-body predictions
