import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from rb2301_ca1.obstacle_generator import (
    generate_sdf_file,
    resolve_coke_model_uris,
)


class ObstacleGeneratorTests(unittest.TestCase):
    def test_generation_preserves_non_coke_content_and_builds_full_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            world_path = Path(directory) / "world.sdf"
            world_path.write_text(
                "<sdf version='1.10'><world name='empty'>"
                "<model name='wall'/><include><name>other</name></include>"
                "<include><name>coke_old</name></include>"
                "</world></sdf>",
                encoding="utf-8",
            )
            active_count = generate_sdf_file(world_path, "../meshes/coke/6", seed=2301)
            world = ET.parse(world_path).getroot().find("world")
            names = [element.findtext("name") for element in world.findall("include")]

            self.assertGreaterEqual(active_count, 32)
            self.assertIn("other", names)
            self.assertNotIn("coke_old", names)
            self.assertEqual(names[-64:], [f"coke{i}" for i in range(1, 65)])
            self.assertEqual(world.findall("include")[-1].findtext("uri"), "../meshes/coke/6")

    def test_absolute_model_directory_becomes_file_uri(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world_path = root / "world.sdf"
            model_directory = root / "Coke model"
            model_directory.mkdir()
            world_path.write_text(
                "<sdf version='1.10'><world name='empty'>"
                "<include><uri>bad-relative-uri</uri><name>coke1</name>"
                "<pose>1 2 0 0 0 0</pose></include>"
                "</world></sdf>",
                encoding="utf-8",
            )

            updated = resolve_coke_model_uris(world_path, model_directory)
            include = ET.parse(world_path).getroot().find("world/include")

            self.assertEqual(updated, 1)
            self.assertEqual(include.findtext("uri"), model_directory.as_uri())
            self.assertEqual(include.findtext("pose"), "1 2 0 0 0 0")


if __name__ == "__main__":
    unittest.main()
