import matplotlib

# Use a non-interactive backend suitable for testing.
matplotlib.use("Agg")

# Ensure project root is on sys.path for direct test execution.
from visualization.cli import create_parser


def test_create_parser_parses_basic_flags():
    parser = create_parser()
    args = parser.parse_args(["meshes/cube.json", "--no-facets", "--scatter", "--tilt"])

    assert args.input == "meshes/cube.json"
    assert args.no_facets is True
    assert args.scatter is True
    assert args.tilt is True
