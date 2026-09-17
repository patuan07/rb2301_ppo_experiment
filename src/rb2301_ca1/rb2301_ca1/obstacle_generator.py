"""Populate the Gazebo SDF with a fixed pool of randomized Coke-can models."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from .maze_layout import (
    DIFFICULTIES,
    LayoutConfig,
    generate_layout,
    layout_config_for_difficulty,
)


def model_uri(model_directory: str | Path) -> str:
    """Return an SDF-compatible URI for a model directory."""

    model_text = str(model_directory)
    if "://" in model_text or not Path(model_text).is_absolute():
        return model_text
    return Path(model_text).resolve().as_uri()


def resolve_coke_model_uris(
    world_file: str | Path,
    model_directory: str | Path,
) -> int:
    """Make existing Coke includes portable at runtime and preserve their poses."""

    path = Path(world_file)
    tree = ET.parse(path)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"No <world> element found in {path}")

    resolved_uri = model_uri(model_directory)
    updated = 0
    for element in world.findall("include"):
        if not element.findtext("name", default="").startswith("coke"):
            continue
        uri = element.find("uri")
        if uri is None:
            uri = ET.SubElement(element, "uri")
        uri.text = resolved_uri
        updated += 1

    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return updated


def add_coke_element(
    model_uri: str,
    x: float,
    y: float,
    z: float,
    number: int,
) -> ET.Element:
    obstacle = ET.Element("include")
    ET.SubElement(obstacle, "uri").text = model_uri
    ET.SubElement(obstacle, "name").text = f"coke{number}"
    ET.SubElement(obstacle, "pose").text = f"{x:.4f} {y:.4f} {z:.4f} 0 0 0"
    return obstacle


def generate_sdf_file(
    world_file: str | Path,
    model_directory: str | Path,
    seed: int | None = None,
    layout_config: LayoutConfig = LayoutConfig(),
) -> int:
    """Replace Coke includes and return the number of active obstacles.

    Exactly 64 named can entities are created. Unused entities are parked below
    the world so the RL environment can randomize every episode efficiently via
    Gazebo's set_pose_vector service.
    """

    path = Path(world_file)
    resolved_model_uri = model_uri(model_directory)
    layout = generate_layout(seed, layout_config)
    tree = ET.parse(path)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"No <world> element found in {path}")

    for element in list(world):
        if element.tag != "include":
            continue
        name = element.findtext("name", default="")
        if name.startswith("coke"):
            world.remove(element)

    for number, (x, y, z) in enumerate(layout.all_positions, start=1):
        world.append(add_coke_element(resolved_model_uri, x, y, z, number))

    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(
        f"Generated {len(layout.active_positions)} active obstacles "
        f"({len(layout.all_positions)} pooled models, "
        f"path={layout.path_length:.2f} m, blockers={layout.straight_blockers}, "
        f"attempt={layout.generation_attempt}) in {path}"
    )
    return len(layout.active_positions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_file", type=Path)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--maze-difficulty", choices=DIFFICULTIES, default="medium")
    args = parser.parse_args()
    generate_sdf_file(
        args.world_file,
        args.model_directory,
        args.seed,
        layout_config_for_difficulty(args.maze_difficulty),
    )


if __name__ == "__main__":
    main()
