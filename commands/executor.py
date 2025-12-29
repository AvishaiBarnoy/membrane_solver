"""Command execution helpers for interactive and scripted runs."""

from __future__ import annotations

import logging
from typing import Iterable

from commands.registry import get_command

logger = logging.getLogger("membrane_solver")


def execute_command_line(
    context,
    line: str,
    *,
    get_command_fn=get_command,
    macro_stack: tuple[str, ...] = (),
    max_macro_depth: int = 20,
) -> None:
    """Execute one command line, expanding any matching macro.

    Macros are defined on ``context.mesh.macros`` as name -> list of command
    strings (each string is fed back through this same executor).
    """
    line = (line or "").strip()
    if not line:
        return

    parts = line.split()
    cmd_name = parts[0]
    cmd_args = parts[1:]

    command, extra_args = get_command_fn(cmd_name)
    if command is not None:
        command.execute(context, extra_args + cmd_args)
        history = getattr(context, "history", None)
        if history is not None:
            history.append(line)
        return

    macros = getattr(context.mesh, "macros", {}) or {}
    if cmd_name in macros:
        if cmd_args:
            logger.warning(
                "Macro '%s' does not accept arguments; ignoring %s", cmd_name, cmd_args
            )

        if len(macro_stack) >= max_macro_depth:
            raise RuntimeError(
                f"Macro expansion exceeded max depth ({max_macro_depth}): {' -> '.join(macro_stack + (cmd_name,))}"
            )
        if cmd_name in macro_stack:
            raise RuntimeError(
                f"Recursive macro call detected: {' -> '.join(macro_stack + (cmd_name,))}"
            )

        for macro_line in _iter_macro_lines(macros[cmd_name]):
            execute_command_line(
                context,
                macro_line,
                macro_stack=macro_stack + (cmd_name,),
                max_macro_depth=max_macro_depth,
            )
        return

    logger.warning("Unknown instruction: %s", cmd_name)


def _iter_macro_lines(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        line = (line or "").strip()
        if line:
            yield line
