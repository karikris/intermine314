"""Read authoritative release metadata without importing the source package."""

import ast
import tomllib


def project_version(root):
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if not isinstance(version, str) or not version:
        raise RuntimeError("Project version must be a non-empty string")
    tree = ast.parse((root / "src/intermine314/_version.py").read_text())
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    runtime = ast.literal_eval(assignments.get("VERSION"))
    alias = assignments.get("__version__")
    public = runtime if isinstance(alias, ast.Name) and alias.id == "VERSION" else ast.literal_eval(alias)
    if runtime != version or public != version:
        raise RuntimeError(f"Project/runtime versions differ: project={version}, VERSION={runtime}, __version__={public}")
    return version
