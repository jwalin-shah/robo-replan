import json
import unittest
from pathlib import Path

from scripts.plot_curriculum import _parse_episode_row


class PlotCurriculumParserTest(unittest.TestCase):
    def test_parse_episode_row_preserves_valid_fields(self):
        row = _parse_episode_row(
            json.dumps(
                {
                    "episode_id": 7,
                    "success": True,
                    "total_reward": 3,
                    "total_steps": 4,
                    "difficulty": "hard",
                }
            ),
            line_number=2,
            path=Path("fixture.jsonl"),
        )

        self.assertEqual(
            row,
            {
                "episode_id": 7,
                "success": 1,
                "total_reward": 3.0,
                "total_steps": 4,
                "difficulty": "hard",
            },
        )

    def test_parse_episode_row_rejects_missing_required_field(self):
        with self.assertRaisesRegex(
            ValueError,
            r"fixture\.jsonl:5 missing required field 'total_reward'",
        ):
            _parse_episode_row(
                json.dumps(
                    {
                        "episode_id": 7,
                        "success": True,
                        "total_steps": 4,
                        "difficulty": "hard",
                    }
                ),
                line_number=5,
                path=Path("fixture.jsonl"),
            )

    def test_parse_episode_row_rejects_wrong_field_type(self):
        with self.assertRaisesRegex(
            ValueError,
            r"fixture\.jsonl:6 field 'success' must be bool",
        ):
            _parse_episode_row(
                json.dumps(
                    {
                        "episode_id": 7,
                        "success": "true",
                        "total_reward": 3.0,
                        "total_steps": 4,
                        "difficulty": "hard",
                    }
                ),
                line_number=6,
                path=Path("fixture.jsonl"),
            )

    def test_parse_episode_row_rejects_bool_for_numeric_field(self):
        with self.assertRaisesRegex(
            ValueError,
            r"fixture\.jsonl:7 field 'total_steps' must be int",
        ):
            _parse_episode_row(
                json.dumps(
                    {
                        "episode_id": 7,
                        "success": True,
                        "total_reward": 3.0,
                        "total_steps": False,
                        "difficulty": "hard",
                    }
                ),
                line_number=7,
                path=Path("fixture.jsonl"),
            )


if __name__ == "__main__":
    unittest.main()
