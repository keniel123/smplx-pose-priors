# SMPL-X Pose Prior Models

A comprehensive implementation of pose prior models for SMPL-X using PyTorch Lightning, including Gaussian priors, per-joint conditional normalizing flows, and global conditional flows.

## 🎯 Overview

This project implements three different types of pose priors for SMPL-X (55 joints):

1. **Gaussian Prior** - Simple factorized Gaussian distributions per joint
2. **Joint Limit Flow** - Per-joint conditional normalizing flows with kinematic conditioning
3. **Global Conditional Flow** - Full pose conditional normalizing flows with enhanced stability

All models are implemented with professional PyTorch Lightning training infrastructure.

## 🏗️ Architecture

### Models

- **`JointLimitGaussian`** - Factorized Gaussian prior with MLE fitting
- **`JointLimitFlow`** - Per-joint conditional RealNVP flows (55 joints)
- **`CondFlowNet`** - Global conditional flow with ActNorm and stability features

### Training Infrastructure

- **`gaussian_trainer.py`** - Lightning trainer for Gaussian priors
- **`joint_flow_trainer.py`** - Lightning trainer for joint limit flows
- **`global_flow_trainer.py`** - Lightning trainer for global conditional flows

### Utilities

- **`rotation_utils.py`** - Rotation representations (6D, axis-angle, matrices)
- **`debug_coupling.py`** - Debugging tools for coupling layers

## 🚀 Quick Start

### Requirements

```bash
pip install torch pytorch-lightning tensorboard
```

### Training Examples

**Gaussian Prior:**
```python
python gaussian_trainer.py
```

**Joint Limit Flows:**
```python
python joint_flow_trainer.py
```

**Global Conditional Flows:**
```python
python global_flow_trainer.py
```

## 📊 Model Performance

### Gaussian Prior
- **Parameters**: 330 (55 joints × 3 dims × 2 params)
- **Training**: Very fast convergence with MLE
- **Use case**: Simple baseline, fast inference

### Joint Limit Flows
- **Parameters**: 517K (55 × per-joint flows)
- **Training**: NLL: 158.9 → 40.4, BPD: 76.4 → 19.4
- **Use case**: Kinematic conditioning, per-joint distributions

### Global Conditional Flows
- **Parameters**: 343K (6 layers, 512 hidden)
- **Training**: NLL: 64.4 → 44.4 over 3 epochs
- **Use case**: Full pose modeling with global dependencies

## 🔧 Technical Features

### SMPL-X Integration
- **55 joints** including eye joints (set to zeros)
- **Real kinematic tree** with proper parent relationships
- **Axis-angle representation** with angle wrapping

### Normalizing Flows
- **RealNVP architecture** with affine coupling layers
- **Conditional flows** with parent joint conditioning
- **ActNorm layers** for training stability
- **6D rotation representation** for continuous optimization

### PyTorch Lightning Benefits
- ✅ **Automatic GPU/multi-GPU** handling
- ✅ **Professional logging** (TensorBoard, W&B)
- ✅ **Checkpointing** and model restoration
- ✅ **Early stopping** and LR scheduling
- ✅ **Gradient clipping** and mixed precision
- ✅ **Distributed training** ready

## 📁 Project Structure

```
├── POSE_PRIORS_README.md
├── .gitignore
│
├── Models/
│   ├── joint_limit_gaussian.py    # Factorized Gaussian prior
│   ├── joint_limit_flow.py        # Per-joint conditional flows
│   └── global_conditional_flow.py # Global conditional flow
│
├── Training/
│   ├── gaussian_trainer.py        # Lightning trainer for Gaussian
│   ├── joint_flow_trainer.py      # Lightning trainer for joint flows
│   └── global_flow_trainer.py     # Lightning trainer for global flow
│
├── Utils/
│   ├── rotation_utils.py          # Rotation utilities
│   └── debug_coupling.py          # Debugging tools
│
└── Tests/
    └── (Generated during training)
```

## 🧪 Testing

Each trainer includes comprehensive testing:

- **Data preprocessing** validation
- **Forward/backward** pass testing
- **Sampling** and generation testing
- **Round-trip** reconstruction testing
- **Checkpoint** loading/saving testing

## 🎛️ Configuration

### Hyperparameters

**Gaussian Prior:**
- Learning rate: 1e-2
- Scheduler: Step LR
- Very fast convergence

**Joint Flows:**
- Learning rate: 1e-3
- Weight decay: 1e-5
- Gradient clipping: 1.0
- Coupling layers: 4 per joint

**Global Flow:**
- Learning rate: 1e-3
- Scheduler: Cosine annealing
- ActNorm: Enabled
- Coupling layers: 6 global

## 📈 Validation Metrics

- **NLL (Negative Log-Likelihood)** - Primary training objective
- **BPD (Bits Per Dimension)** - Normalized likelihood metric
- **Round-trip MSE** - Reconstruction accuracy
- **Sampling statistics** - Generated sample quality
- **Per-joint analysis** - Individual joint performance

## 🔄 SMPL-X Joint Structure

The models handle the complete SMPL-X joint hierarchy:
- **Joints 0-21**: Global orient + body pose (22 joints)
- **Joint 22**: Jaw pose (1 joint)
- **Joints 23-24**: Eye poses (2 joints, set to zeros)
- **Joints 25-39**: Left hand pose (15 joints)
- **Joints 40-54**: Right hand pose (15 joints)

Total: **55 joints** with proper kinematic parent relationships

## 🚀 Advanced Features

### Rotation Representations
- **6D continuous representation** for optimization
- **Axis-angle** for SMPL-X compatibility
- **Rotation matrices** with orthogonality constraints
- **Angle wrapping** to [-π, π] for stability

### Flow Architecture
- **Conditional coupling layers** with parent conditioning
- **ActNorm normalization** for stable training
- **Fixed permutations** for reproducibility
- **Gradient clipping** for numerical stability

### Professional Training
- **Multi-GPU ready** with Lightning DDP
- **Automatic mixed precision** for faster training
- **Comprehensive logging** with experiment tracking
- **Robust checkpointing** with best model selection

## 📝 Citation

```bibtex
@misc{smplx-pose-priors,
  title={SMPL-X Pose Prior Models with PyTorch Lightning},
  author={Keniel Peart},
  year={2024},
  howpublished={\url{https://github.com/keniel123/smplx-pose-priors}}
}
```

## 🤝 Contributing

Contributions welcome! Please check issues and submit PRs.

## 📄 License

MIT License - see LICENSE file for details.

---

🔬 **Generated with [Claude Code](https://claude.ai/code)**