#!/usr/bin/env python3
"""
Create Simple Skeleton OBJ for T-Pose Visualization

Creates a basic stick figure skeleton representation of the SMPL-X T-pose
for visualization purposes without requiring the full SMPL-X model.
"""

import numpy as np
from pathlib import Path
import argparse


def create_skeleton_tpose_obj(output_dir: str = 'tpose_reference'):
    """
    Create a simple skeleton OBJ file representing SMPL-X T-pose structure.

    Args:
        output_dir: Output directory for the OBJ file
    """

    print(f"🔧 Creating skeleton T-pose OBJ...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Define approximate joint positions for T-pose (in meters)
    # These are rough estimates for visualization purposes
    joint_positions = {
        # Core body
        'pelvis': [0.0, 0.0, 0.0],           # Root/origin
        'spine1': [0.0, 0.1, 0.0],
        'spine2': [0.0, 0.25, 0.0],
        'spine3': [0.0, 0.4, 0.0],
        'neck': [0.0, 0.55, 0.0],
        'head': [0.0, 0.65, 0.0],

        # Arms (extended horizontally for T-pose)
        'l_collar': [-0.05, 0.5, 0.0],
        'r_collar': [0.05, 0.5, 0.0],
        'l_shoulder': [-0.15, 0.45, 0.0],
        'r_shoulder': [0.15, 0.45, 0.0],
        'l_elbow': [-0.4, 0.45, 0.0],
        'r_elbow': [0.4, 0.45, 0.0],
        'l_wrist': [-0.65, 0.45, 0.0],
        'r_wrist': [0.65, 0.45, 0.0],

        # Legs
        'l_hip': [-0.1, -0.05, 0.0],
        'r_hip': [0.1, -0.05, 0.0],
        'l_knee': [-0.1, -0.4, 0.0],
        'r_knee': [0.1, -0.4, 0.0],
        'l_ankle': [-0.1, -0.8, 0.0],
        'r_ankle': [0.1, -0.8, 0.0],
        'l_foot': [-0.1, -0.85, 0.1],
        'r_foot': [0.1, -0.85, 0.1],

        # Face
        'jaw': [0.0, 0.6, 0.05],
        'l_eye': [-0.03, 0.65, 0.08],
        'r_eye': [0.03, 0.65, 0.08],

        # Left hand (simplified)
        'l_thumb1': [-0.68, 0.47, 0.02],
        'l_thumb2': [-0.7, 0.49, 0.04],
        'l_thumb3': [-0.72, 0.5, 0.05],
        'l_index1': [-0.7, 0.48, 0.0],
        'l_index2': [-0.73, 0.48, 0.0],
        'l_index3': [-0.76, 0.48, 0.0],
        'l_middle1': [-0.7, 0.45, 0.0],
        'l_middle2': [-0.73, 0.45, 0.0],
        'l_middle3': [-0.76, 0.45, 0.0],
        'l_ring1': [-0.7, 0.42, 0.0],
        'l_ring2': [-0.73, 0.42, 0.0],
        'l_ring3': [-0.76, 0.42, 0.0],
        'l_pinky1': [-0.7, 0.39, 0.0],
        'l_pinky2': [-0.73, 0.39, 0.0],
        'l_pinky3': [-0.76, 0.39, 0.0],

        # Right hand (mirrored)
        'r_thumb1': [0.68, 0.47, 0.02],
        'r_thumb2': [0.7, 0.49, 0.04],
        'r_thumb3': [0.72, 0.5, 0.05],
        'r_index1': [0.7, 0.48, 0.0],
        'r_index2': [0.73, 0.48, 0.0],
        'r_index3': [0.76, 0.48, 0.0],
        'r_middle1': [0.7, 0.45, 0.0],
        'r_middle2': [0.73, 0.45, 0.0],
        'r_middle3': [0.76, 0.45, 0.0],
        'r_ring1': [0.7, 0.42, 0.0],
        'r_ring2': [0.73, 0.42, 0.0],
        'r_ring3': [0.76, 0.42, 0.0],
        'r_pinky1': [0.7, 0.39, 0.0],
        'r_pinky2': [0.73, 0.39, 0.0],
        'r_pinky3': [0.76, 0.39, 0.0],
    }

    # Define skeleton connections (bones)
    skeleton_bones = [
        # Main body chain
        ('pelvis', 'spine1'),
        ('spine1', 'spine2'),
        ('spine2', 'spine3'),
        ('spine3', 'neck'),
        ('neck', 'head'),

        # Arms
        ('spine3', 'l_collar'),
        ('spine3', 'r_collar'),
        ('l_collar', 'l_shoulder'),
        ('r_collar', 'r_shoulder'),
        ('l_shoulder', 'l_elbow'),
        ('r_shoulder', 'r_elbow'),
        ('l_elbow', 'l_wrist'),
        ('r_elbow', 'r_wrist'),

        # Legs
        ('pelvis', 'l_hip'),
        ('pelvis', 'r_hip'),
        ('l_hip', 'l_knee'),
        ('r_hip', 'r_knee'),
        ('l_knee', 'l_ankle'),
        ('r_knee', 'r_ankle'),
        ('l_ankle', 'l_foot'),
        ('r_ankle', 'r_foot'),

        # Face
        ('head', 'jaw'),
        ('head', 'l_eye'),
        ('head', 'r_eye'),

        # Left hand
        ('l_wrist', 'l_thumb1'),
        ('l_thumb1', 'l_thumb2'),
        ('l_thumb2', 'l_thumb3'),
        ('l_wrist', 'l_index1'),
        ('l_index1', 'l_index2'),
        ('l_index2', 'l_index3'),
        ('l_wrist', 'l_middle1'),
        ('l_middle1', 'l_middle2'),
        ('l_middle2', 'l_middle3'),
        ('l_wrist', 'l_ring1'),
        ('l_ring1', 'l_ring2'),
        ('l_ring2', 'l_ring3'),
        ('l_wrist', 'l_pinky1'),
        ('l_pinky1', 'l_pinky2'),
        ('l_pinky2', 'l_pinky3'),

        # Right hand
        ('r_wrist', 'r_thumb1'),
        ('r_thumb1', 'r_thumb2'),
        ('r_thumb2', 'r_thumb3'),
        ('r_wrist', 'r_index1'),
        ('r_index1', 'r_index2'),
        ('r_index2', 'r_index3'),
        ('r_wrist', 'r_middle1'),
        ('r_middle1', 'r_middle2'),
        ('r_middle2', 'r_middle3'),
        ('r_wrist', 'r_ring1'),
        ('r_ring1', 'r_ring2'),
        ('r_ring2', 'r_ring3'),
        ('r_wrist', 'r_pinky1'),
        ('r_pinky1', 'r_pinky2'),
        ('r_pinky2', 'r_pinky3'),
    ]

    # Create vertices list and line indices
    vertices = []
    vertex_map = {}

    # Add all joint positions as vertices
    for i, (joint_name, pos) in enumerate(joint_positions.items()):
        vertices.append(pos)
        vertex_map[joint_name] = i

    print(f"📊 Skeleton structure:")
    print(f"  Joints: {len(vertices)}")
    print(f"  Bones: {len(skeleton_bones)}")

    # Create OBJ file
    obj_file = output_path / "skeleton_tpose.obj"

    with open(obj_file, 'w') as f:
        f.write("# SMPL-X T-Pose Skeleton\n")
        f.write("# Simple stick figure representation\n")
        f.write(f"# Generated with {len(vertices)} joints and {len(skeleton_bones)} bones\n\n")

        # Write vertices
        f.write("# Vertices (joint positions)\n")
        for i, pos in enumerate(vertices):
            joint_name = list(joint_positions.keys())[i]
            f.write(f"v {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}  # {joint_name}\n")

        f.write("\n# Lines (bones)\n")
        # Write lines (bones)
        for bone in skeleton_bones:
            joint1, joint2 = bone
            if joint1 in vertex_map and joint2 in vertex_map:
                v1 = vertex_map[joint1] + 1  # OBJ uses 1-based indexing
                v2 = vertex_map[joint2] + 1
                f.write(f"l {v1} {v2}  # {joint1} -> {joint2}\n")

    print(f"💾 Skeleton OBJ saved: {obj_file}")

    # Create a PLY file as well (better for some viewers)
    ply_file = output_path / "skeleton_tpose.ply"
    create_ply_skeleton(vertices, skeleton_bones, vertex_map, joint_positions, ply_file)

    # Create visualization info
    info_file = output_path / "skeleton_info.txt"
    with open(info_file, 'w') as f:
        f.write("SMPL-X Skeleton T-Pose Information\n")
        f.write("="*50 + "\n\n")

        f.write("Files created:\n")
        f.write(f"- {obj_file.name}: OBJ format (lines)\n")
        f.write(f"- {ply_file.name}: PLY format (lines)\n\n")

        f.write("Viewing instructions:\n")
        f.write("- Blender: File > Import > Wavefront (.obj)\n")
        f.write("- MeshLab: File > Import Mesh\n")
        f.write("- Online viewers: threejs.org/editor\n\n")

        f.write("Coordinate system:\n")
        f.write("- Y-axis: Up (positive = up)\n")
        f.write("- X-axis: Right (positive = right)\n")
        f.write("- Z-axis: Forward (positive = forward)\n\n")

        f.write("Scale: Approximately 1.7 meters tall human\n")
        f.write("Pose: T-pose with arms extended horizontally\n")

    print(f"💾 PLY file saved: {ply_file}")
    print(f"📄 Info file saved: {info_file}")

    return True


def create_ply_skeleton(vertices, skeleton_bones, vertex_map, joint_positions, ply_file):
    """Create PLY format skeleton file"""

    with open(ply_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(skeleton_bones)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        # Write vertices
        for pos in vertices:
            f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

        # Write edges
        for bone in skeleton_bones:
            joint1, joint2 = bone
            if joint1 in vertex_map and joint2 in vertex_map:
                v1 = vertex_map[joint1]  # PLY uses 0-based indexing
                v2 = vertex_map[joint2]
                f.write(f"{v1} {v2}\n")


def main():
    parser = argparse.ArgumentParser(description='Create skeleton OBJ for T-pose visualization')
    parser.add_argument('--output-dir', type=str, default='tpose_reference',
                       help='Output directory')

    args = parser.parse_args()

    success = create_skeleton_tpose_obj(output_dir=args.output_dir)

    if success:
        print(f"\n🎉 Skeleton T-pose OBJ generation complete!")
        print(f"💡 Open the OBJ/PLY file in Blender, MeshLab, or online 3D viewer")
        print(f"💡 This shows the basic joint structure of SMPL-X in T-pose")
    else:
        print(f"\n❌ Skeleton generation failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())