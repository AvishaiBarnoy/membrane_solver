# Debugging Guide

To help you debug the `membrane_solver` project effectively, we have implemented a "standard debugging" workflow for CLI applications.

## 1. Post-Mortem Debugging (`--debugger`)

`main.py` supports a `--debugger` flag that enters a post-mortem debugger on uncaught exceptions. It prefers `ipdb` and falls back to `pdb`.

*   **How to use:**
    Run your command with the `--debugger` flag. If the program crashes (raises an uncaught exception), it will automatically drop you into an interactive `ipdb` (or `pdb`) shell at the point of failure.
    ```bash
    python main.py -i meshes/cube.json --debugger
    ```
*   **Why use it:** This is the fastest way to understand *why* a crash happened. You can inspect variables (`p variable_name`), move up the stack (`u`), and see the exact state of the application when it died.

## 2. Verbose Logging (`--debug`)

The project supports structured logging.

*   **How to use:**
    Add the `--debug` flag to enable verbose output.
    ```bash
    python main.py -i meshes/cube.json --debug
    ```
*   **Where to look:**
    Logs are written to `membrane_solver.log` in the current directory. You can `tail -f membrane_solver.log` in a separate terminal window to watch the logs in real-time.

## 3. Manual Breakpoints

If you need to stop execution at a specific line *before* a crash:

1.  **Code:** Insert this line in your python code where you want to stop:
    ```python
    import pdb; pdb.set_trace()
    ```
    (Or `import ipdb; ipdb.set_trace()` if you have it installed).
2.  **Run:** Execute the program normally. It will pause at that line.

## 4. Interactive Debugger Cheat Sheet

Using the debugger (like `pdb` or `ipdb`) feels like pausing a video game to look around the map. When your program stops—either because it crashed with `--debugger` or hit a line where you put `pdb.set_trace()`—you enter a special interactive shell.

Here are the essential commands to navigate this shell:

### See Where You Are
*   **`l` (list)**: Shows the code around the line where the program paused. The arrow `->` points to the line that is *about to run*.
*   **`w` (where)**: Prints the "stack trace." It tells you exactly which function you are in, and which function called that one, all the way back to `main`.

### Inspect Variables
You can type almost any Python code to check values.
*   **`p variable_name`**: Prints the value of a variable.
    *   *Example:* `p mesh.vertices[0]`
*   **`whatis variable_name`**: Tells you the type of the object (e.g., `list`, `int`, `Mesh`).

### Move Through Code
*   **`n` (next)**: Runs the current line and stops at the next one. Use this to step over lines one by one.
*   **`s` (step)**: Steps **inside** a function call. Use this if the current line calls a function and you want to see what happens *inside* that function.
*   **`c` (continue)**: Unpauses the program and lets it run until it finishes, crashes again, or hits another breakpoint.

### Move Up and Down Functions
Sometimes you crash inside a helper function (like `compute_volume`), but the real bad data came from the function that called it.
*   **`u` (up)**: Move "up" the stack to the function that called the current one. This lets you inspect variables in the *caller's* scope.
*   **`d` (down)**: Move back "down" into the function you were just looking at.

### Exiting
*   **`q` (quit)**: Aborts the program immediately.

### Summary Workflow
1.  **Crash:** Program stops, you see the `ipdb>` prompt.
2.  **Look:** Type `l` to see the code.
3.  **Inspect:** Type `p my_var` to check if a variable is `None` or has the wrong value.
4.  **Trace:** If the variable is wrong, type `u` (up) to see where it came from in the previous function.
5.  **Quit:** Type `q` to exit when you've found the bug.

## 5. VS Code Configuration (Optional)

If you use VS Code, you can create a `.vscode/launch.json` file to debug visually. Here is a standard configuration for the project:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Membrane Solver (Debug)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": ["-i", "${workspaceFolder}/meshes/cube.json", "--debug"],
            "console": "integratedTerminal"
        }
    ]
}
```
