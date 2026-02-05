import pathlib


def test_tests_do_not_depend_on_meshes_caveolin_inputs() -> None:
    """Guard against brittle tests that load user-editable meshes/caveolin YAMLs."""
    root = pathlib.Path(__file__).resolve().parent
    offenders: list[str] = []
    needle = "meshes/" + "caveolin"

    for path in root.rglob("test_*.py"):
        if path.name == "test_no_meshes_caveolin_dependency_regression.py":
            continue
        text = path.read_text(encoding="utf-8")
        if needle in text:
            offenders.append(str(path.relative_to(root)))

    assert not offenders, "Tests must not load from meshes/caveolin: " + ", ".join(
        sorted(offenders)
    )
