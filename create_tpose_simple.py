#!/usr/bin/env python3
"""
Simple SMPL-X T-Pose Generator

Creates the canonical T-pose pose parameters without requiring SMPL-X model files.
Generates the pose vector in our dataset format (55 joints * 3 = 165D).
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json


def create_tpose_parameters(output_dir: str = 'tpose_reference'):
    """
    Create SMPL-X T-pose parameters (all zeros) and save them.

    Args:
        output_dir: Output directory for generated files
    """

    print(f"🔧 Creating SMPL-X T-pose parameters...")
    print(f"  Output dir: {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # SMPL-X T-pose: all joint angles are zero (identity rotations)
    batch_size = 1

    # Individual components (SMPL-X format)
    global_orient = torch.zeros(batch_size, 3)      # Global orientation (root)
    body_pose = torch.zeros(batch_size, 21 * 3)     # Body joints (21 joints)
    jaw_pose = torch.zeros(batch_size, 3)           # Jaw
    leye_pose = torch.zeros(batch_size, 3)          # Left eye
    reye_pose = torch.zeros(batch_size, 3)          # Right eye
    left_hand_pose = torch.zeros(batch_size, 15 * 3)   # Left hand (15 joints)
    right_hand_pose = torch.zeros(batch_size, 15 * 3)  # Right hand (15 joints)

    # Shape and expression parameters (neutral)
    betas = torch.zeros(batch_size, 10)              # Shape parameters
    expression = torch.zeros(batch_size, 10)         # Expression parameters

    print(f"📊 T-pose parameters:")
    print(f"  Global orient: {global_orient.shape} (all zeros)")
    print(f"  Body pose: {body_pose.shape} (all zeros)")
    print(f"  Jaw pose: {jaw_pose.shape} (all zeros)")
    print(f"  Eye poses: {leye_pose.shape} each (all zeros)")
    print(f"  Hand poses: {left_hand_pose.shape} each (all zeros)")

    # Create dataset format pose (55 joints * 3 = 165D)
    # Order: [global(1), body(21), jaw(1), leye(1), reye(1), lhand(15), rhand(15)]
    dataset_pose = torch.cat([
        global_orient.reshape(1, 3),          # 1 joint * 3 = 3D
        body_pose.reshape(1, 21 * 3),         # 21 joints * 3 = 63D
        jaw_pose.reshape(1, 3),               # 1 joint * 3 = 3D
        leye_pose.reshape(1, 3),              # 1 joint * 3 = 3D
        reye_pose.reshape(1, 3),              # 1 joint * 3 = 3D
        left_hand_pose.reshape(1, 15 * 3),    # 15 joints * 3 = 45D
        right_hand_pose.reshape(1, 15 * 3)    # 15 joints * 3 = 45D
    ], dim=1)  # Total: 165D

    print(f"  Dataset format: {dataset_pose.shape} (55 joints * 3)")

    # Convert to numpy
    dataset_pose_np = dataset_pose.numpy()

    # Save SMPL-X format parameters
    smplx_file = output_path / "tpose_smplx_params.npz"
    np.savez(
        smplx_file,
        global_orient=global_orient.numpy(),
        body_pose=body_pose.numpy(),
        jaw_pose=jaw_pose.numpy(),
        leye_pose=leye_pose.numpy(),
        reye_pose=reye_pose.numpy(),
        left_hand_pose=left_hand_pose.numpy(),
        right_hand_pose=right_hand_pose.numpy(),
        betas=betas.numpy(),
        expression=expression.numpy()
    )

    # Save dataset format
    dataset_file = output_path / "tpose_dataset_format.npz"
    np.savez(dataset_file, pose=dataset_pose_np)

    # Save as text for easy viewing
    pose_txt_file = output_path / "tpose_pose_vector.txt"
    with open(pose_txt_file, 'w') as f:
        f.write("SMPL-X T-Pose (55 joints * 3 = 165 parameters)\n")
        f.write("All values are zero (identity rotations)\n\n")
        f.write("Joint order:\n")
        f.write("0: Global orientation (3)\n")
        f.write("1-21: Body joints (63)\n")
        f.write("22: Jaw (3)\n")
        f.write("23: Left eye (3)\n")
        f.write("24: Right eye (3)\n")
        f.write("25-39: Left hand (45)\n")
        f.write("40-54: Right hand (45)\n\n")

        # Write pose vector reshaped as (55, 3)
        pose_55x3 = dataset_pose_np.reshape(55, 3)
        f.write("Pose vector (55 joints, 3 dims each):\n")
        for i, joint in enumerate(pose_55x3):
            joint_name = get_joint_name(i)
            f.write(f"Joint {i:2d} ({joint_name:12s}): [{joint[0]:8.5f}, {joint[1]:8.5f}, {joint[2]:8.5f}]\n")

    # Save joint mapping
    joint_mapping_file = output_path / "joint_mapping.json"
    joint_mapping = {}
    for i in range(55):
        joint_mapping[i] = {
            "name": get_joint_name(i),
            "index": i,
            "type": get_joint_type(i)
        }

    with open(joint_mapping_file, 'w') as f:
        json.dump(joint_mapping, f, indent=2)

    # Create simple visualization data
    create_simple_visualization(dataset_pose_np, output_path)

    print(f"\n📁 Files saved:")
    print(f"  📊 SMPL-X params: {smplx_file}")
    print(f"  📋 Dataset format: {dataset_file}")
    print(f"  📝 Text format: {pose_txt_file}")
    print(f"  🗂️  Joint mapping: {joint_mapping_file}")

    return True


def get_joint_name(index):
    """Get human-readable joint name"""
    joint_names = [
        "global",         # 0
        "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2",     # 1-7
        "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot", "neck", "l_collar", # 8-14
        "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",   # 15-20
        "l_wrist", "r_wrist",  # 21-22 (last body joints)
        "jaw",            # 22
        "l_eye", "r_eye", # 23-24
        # Left hand joints (25-39)
        "l_thumb1", "l_thumb2", "l_thumb3", "l_index1", "l_index2", "l_index3",
        "l_middle1", "l_middle2", "l_middle3", "l_ring1", "l_ring2", "l_ring3",
        "l_pinky1", "l_pinky2", "l_pinky3",
        # Right hand joints (40-54)
        "r_thumb1", "r_thumb2", "r_thumb3", "r_index1", "r_index2", "r_index3",
        "r_middle1", "r_middle2", "r_middle3", "r_ring1", "r_ring2", "r_ring3",
        "r_pinky1", "r_pinky2", "r_pinky3"
    ]

    if index < len(joint_names):
        return joint_names[index]
    return f"joint_{index}"


def get_joint_type(index):
    """Get joint type"""
    if index == 0:
        return "global"
    elif 1 <= index <= 21:
        return "body"
    elif index == 22:
        return "jaw"
    elif 23 <= index <= 24:
        return "eye"
    elif 25 <= index <= 39:
        return "left_hand"
    elif 40 <= index <= 54:
        return "right_hand"
    else:
        return "unknown"


def create_simple_visualization(pose_vector, output_path):
    """Create a simple text-based visualization of the pose"""
    viz_file = output_path / "tpose_visualization.txt"

    with open(viz_file, 'w') as f:
        f.write("SMPL-X T-Pose Visualization\n")
        f.write("=" * 50 + "\n\n")

        f.write("T-Pose Description:\n")
        f.write("- Person standing upright\n")
        f.write("- Arms extended horizontally to the sides\n")
        f.write("- Legs straight and together\n")
        f.write("- All joint rotations are zero (identity)\n")
        f.write("- Hands are in neutral/flat position\n")
        f.write("- Eyes looking forward\n")
        f.write("- Jaw closed (neutral)\n\n")

        f.write("ASCII Representation:\n")
        f.write("     O    (head)\n")
        f.write("     |    (neck)\n")
        f.write(" ----+----  (arms extended)\n")
        f.write("     |    (spine)\n")
        f.write("     |    (pelvis)\n")
        f.write("    / \\   (legs)\n")
        f.write("   /   \\  \n\n")

        f.write("Joint Angles (all zero for T-pose):\n")
        pose_55x3 = pose_vector.reshape(55, 3)

        # Group by type
        f.write("\nGlobal Orientation:\n")
        f.write(f"  {get_joint_name(0)}: {pose_55x3[0]}\n")

        f.write("\nBody Joints:\n")
        for i in range(1, 22):
            f.write(f"  {get_joint_name(i):12s}: {pose_55x3[i]}\n")

        f.write("\nFace Joints:\n")
        for i in range(22, 25):
            f.write(f"  {get_joint_name(i):12s}: {pose_55x3[i]}\n")

        f.write("\nLeft Hand Joints:\n")
        for i in range(25, 40):
            f.write(f"  {get_joint_name(i):12s}: {pose_55x3[i]}\n")

        f.write("\nRight Hand Joints:\n")
        for i in range(40, 55):
            f.write(f"  {get_joint_name(i):12s}: {pose_55x3[i]}\n")

    print(f"  📈 Visualization: {viz_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate SMPL-X T-pose parameters (no model required)')
    parser.add_argument('--output-dir', type=str, default='tpose_reference',
                       help='Output directory')

    args = parser.parse_args()

    success = create_tpose_parameters(output_dir=args.output_dir)

    if success:
        print(f"\n🎉 T-pose parameter generation complete!")
        print(f"💡 The T-pose is the canonical pose with all joint angles at zero")
        print(f"💡 Use these parameters as reference for pose analysis")
        print(f"💡 To get actual 3D mesh, you'll need SMPL-X model files from https://smpl-x.is.tue.mpg.de/")
    else:
        print(f"\n❌ T-pose generation failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())