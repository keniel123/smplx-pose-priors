#!/usr/bin/env python3
"""
SMPL-X T-Pose Reference Generator

Creates the canonical T-pose for SMPL-X model and exports it as OBJ file for visualization.
The T-pose has all joint angles set to zero (identity rotations).
"""

import torch
import numpy as np
import smplx
from pathlib import Path
import argparse


def create_smplx_tpose(
    model_path: str = None,
    gender: str = 'neutral',
    output_dir: str = 'tpose_reference',
    use_pca: bool = True,
    num_pca_comps: int = 45
):
    """
    Create SMPL-X T-pose reference and export as OBJ file.

    Args:
        model_path: Path to SMPL-X model files (if None, uses default)
        gender: Model gender ('neutral', 'male', 'female')
        output_dir: Output directory for generated files
        use_pca: Whether to use PCA for hand poses
        num_pca_comps: Number of PCA components for hands
    """

    print(f"🔧 Creating SMPL-X T-pose reference...")
    print(f"  Gender: {gender}")
    print(f"  Use PCA: {use_pca}")
    print(f"  Output dir: {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create SMPL-X model
    try:
        model_kwargs = {
            'gender': gender,
            'use_pca': use_pca,
            'flat_hand_mean': True,  # Start with flat hands
        }

        if model_path:
            model_kwargs['model_path'] = model_path

        if use_pca:
            model_kwargs['num_pca_comps'] = num_pca_comps

        print(f"🤖 Loading SMPL-X model...")
        smplx_model = smplx.create(**model_kwargs)

        print(f"✅ SMPL-X model loaded successfully")

    except Exception as e:
        print(f"❌ Error loading SMPL-X model: {e}")
        print(f"💡 You may need to download SMPL-X model files from: https://smpl-x.is.tue.mpg.de/")
        return False

    # Create T-pose (all zeros)
    batch_size = 1

    # All pose parameters as zeros (T-pose)
    global_orient = torch.zeros(batch_size, 3)      # Global orientation (root)
    body_pose = torch.zeros(batch_size, 21 * 3)     # Body joints (21 joints)
    jaw_pose = torch.zeros(batch_size, 3)           # Jaw
    leye_pose = torch.zeros(batch_size, 3)          # Left eye
    reye_pose = torch.zeros(batch_size, 3)          # Right eye

    # Hand poses - zeros or flat hands
    if use_pca:
        left_hand_pose = torch.zeros(batch_size, num_pca_comps)
        right_hand_pose = torch.zeros(batch_size, num_pca_comps)
    else:
        left_hand_pose = torch.zeros(batch_size, 15 * 3)   # 15 joints * 3 dims
        right_hand_pose = torch.zeros(batch_size, 15 * 3)

    # Shape parameters (average body shape)
    betas = torch.zeros(batch_size, 10)

    # Expression parameters (neutral expression)
    expression = torch.zeros(batch_size, 10)

    print(f"📊 Generating T-pose mesh...")

    # Forward pass through SMPL-X model
    with torch.no_grad():
        output = smplx_model(
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            betas=betas,
            expression=expression,
            return_verts=True
        )

    # Extract vertices and faces
    vertices = output.vertices[0].numpy()  # Shape: (N_verts, 3)
    faces = smplx_model.faces  # Shape: (N_faces, 3)

    print(f"✅ T-pose generated:")
    print(f"  Vertices: {vertices.shape}")
    print(f"  Faces: {faces.shape}")

    # Save as OBJ file
    obj_file = output_path / f"smplx_tpose_{gender}.obj"
    save_obj(vertices, faces, obj_file)

    # Save pose parameters as NPZ
    pose_file = output_path / f"smplx_tpose_{gender}_params.npz"
    save_pose_params(
        global_orient, body_pose, jaw_pose, leye_pose, reye_pose,
        left_hand_pose, right_hand_pose, betas, expression,
        pose_file, use_pca
    )

    # Create pose vector in our dataset format (55 joints * 3 = 165D)
    dataset_pose = create_dataset_format_pose(
        global_orient, body_pose, jaw_pose, leye_pose, reye_pose,
        left_hand_pose, right_hand_pose, use_pca
    )

    pose_dataset_file = output_path / f"smplx_tpose_{gender}_dataset_format.npz"
    np.savez(pose_dataset_file, pose=dataset_pose)

    print(f"\n📁 Files saved:")
    print(f"  🎭 Mesh: {obj_file}")
    print(f"  📊 Parameters: {pose_file}")
    print(f"  📋 Dataset format: {pose_dataset_file}")

    # Print statistics
    print(f"\n📈 T-pose statistics:")
    print(f"  Vertex bounds:")
    print(f"    X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
    print(f"    Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
    print(f"    Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
    print(f"  Dataset pose shape: {dataset_pose.shape}")

    return True


def save_obj(vertices, faces, filename):
    """Save mesh as OBJ file"""
    print(f"💾 Saving OBJ file: {filename}")

    with open(filename, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"✅ OBJ file saved: {filename}")


def save_pose_params(global_orient, body_pose, jaw_pose, leye_pose, reye_pose,
                    left_hand_pose, right_hand_pose, betas, expression,
                    filename, use_pca):
    """Save pose parameters as NPZ file"""
    print(f"💾 Saving pose parameters: {filename}")

    params = {
        'global_orient': global_orient.numpy(),
        'body_pose': body_pose.numpy(),
        'jaw_pose': jaw_pose.numpy(),
        'leye_pose': leye_pose.numpy(),
        'reye_pose': reye_pose.numpy(),
        'left_hand_pose': left_hand_pose.numpy(),
        'right_hand_pose': right_hand_pose.numpy(),
        'betas': betas.numpy(),
        'expression': expression.numpy(),
        'use_pca': use_pca
    }

    np.savez(filename, **params)
    print(f"✅ Pose parameters saved: {filename}")


def create_dataset_format_pose(global_orient, body_pose, jaw_pose, leye_pose, reye_pose,
                              left_hand_pose, right_hand_pose, use_pca):
    """
    Create pose in our dataset format (55 joints * 3 = 165D)
    Order: [global(1), body(21), jaw(1), leye(1), reye(1), lhand(15), rhand(15)]
    """

    # Convert hand poses to 15*3 format if using PCA
    if use_pca:
        # For T-pose with PCA, we use zeros which should give flat hands
        lhand_aa = torch.zeros(1, 15 * 3)  # Will be flat hands
        rhand_aa = torch.zeros(1, 15 * 3)
    else:
        lhand_aa = left_hand_pose
        rhand_aa = right_hand_pose

    # Concatenate in dataset order
    pose_vector = torch.cat([
        global_orient.reshape(1, 3),      # 1 joint * 3 = 3D
        body_pose.reshape(1, 21 * 3),     # 21 joints * 3 = 63D
        jaw_pose.reshape(1, 3),           # 1 joint * 3 = 3D
        leye_pose.reshape(1, 3),          # 1 joint * 3 = 3D
        reye_pose.reshape(1, 3),          # 1 joint * 3 = 3D
        lhand_aa.reshape(1, 15 * 3),      # 15 joints * 3 = 45D
        rhand_aa.reshape(1, 15 * 3)       # 15 joints * 3 = 45D
    ], dim=1)  # Total: 165D

    return pose_vector.numpy()


def main():
    parser = argparse.ArgumentParser(description='Generate SMPL-X T-pose reference')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to SMPL-X model files')
    parser.add_argument('--gender', type=str, default='neutral',
                       choices=['neutral', 'male', 'female'],
                       help='Model gender')
    parser.add_argument('--output-dir', type=str, default='tpose_reference',
                       help='Output directory')
    parser.add_argument('--no-pca', action='store_true',
                       help='Disable PCA for hand poses')
    parser.add_argument('--pca-comps', type=int, default=45,
                       help='Number of PCA components for hands')

    args = parser.parse_args()

    success = create_smplx_tpose(
        model_path=args.model_path,
        gender=args.gender,
        output_dir=args.output_dir,
        use_pca=not args.no_pca,
        num_pca_comps=args.pca_comps
    )

    if success:
        print(f"\n🎉 T-pose reference generation complete!")
        print(f"💡 You can view the OBJ file in any 3D viewer (Blender, MeshLab, etc.)")
    else:
        print(f"\n❌ T-pose generation failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())