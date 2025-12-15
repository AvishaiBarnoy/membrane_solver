import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import parse_instructions


def test_parse_instructions_single_command():
    assert parse_instructions("g5") == ["g5"]
    assert parse_instructions("cg") == ["cg"]
