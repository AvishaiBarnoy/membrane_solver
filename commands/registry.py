from commands.io import PropertiesCommand, SaveCommand, VisualizeCommand
from commands.mesh_ops import (
    EquiangulateCommand,
    PerturbCommand,
    RefineCommand,
    SnapshotCommand,
    VertexAverageCommand,
)
from commands.meta import (
    EnergyCommand,
    HelpCommand,
    HistoryCommand,
    PrintEntityCommand,
    QuitCommand,
    SetCommand,
    StepSizeCommand,
)
from commands.minimization import (
    GoCommand,
    HessianCommand,
    LiveVisCommand,
    SetStepperCommand,
)

COMMAND_REGISTRY = {
    "g": GoCommand(),
    "bfgs": SetStepperCommand("bfgs"),
    "cg": SetStepperCommand("cg"),
    "gd": SetStepperCommand("gd"),
    "hessian": HessianCommand(),
    "lv": LiveVisCommand(),
    "live_vis": LiveVisCommand(),
    "r": RefineCommand(),
    "v": VertexAverageCommand(),
    "vertex_average": VertexAverageCommand(),
    "u": EquiangulateCommand(),
    "perturb": PerturbCommand(),
    "kick": PerturbCommand(),
    "snapshot": SnapshotCommand(),
    "fix": SnapshotCommand(),
    "save": SaveCommand(),
    "s": VisualizeCommand(),
    "visualize": VisualizeCommand(),
    "p": PropertiesCommand(),
    "props": PropertiesCommand(),
    "i": PropertiesCommand(),
    "properties": PropertiesCommand(),
    "q": QuitCommand(),
    "quit": QuitCommand(),
    "exit": QuitCommand(),
    "help": HelpCommand(),
    "h": HelpCommand(),
    "set": SetCommand(),
    "print": PrintEntityCommand(),
    "energy": EnergyCommand(),
    "history": HistoryCommand(),
    "t": StepSizeCommand(),
    "tf": StepSizeCommand(),
}


def get_command(name):
    # Handle g10, r5, etc.
    if name.startswith("g") and name[1:].isdigit():
        return COMMAND_REGISTRY["g"], [name[1:]]
    if name.startswith("r") and name[1:].isdigit():
        return COMMAND_REGISTRY["r"], [name[1:]]
    if name.lower().startswith("v") and name[1:].isdigit():
        return COMMAND_REGISTRY["v"], [name[1:]]
    if name.lower() in {"tf", "tfree"}:
        return COMMAND_REGISTRY["t"], ["free"]
    if name.lower().startswith("t") and len(name) > 1:
        # Allow t1e-3, t0.01, etc.
        return COMMAND_REGISTRY["t"], [name[1:]]

    cmd = COMMAND_REGISTRY.get(name.lower())
    return cmd, []
