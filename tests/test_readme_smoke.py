"""Run the exact public quickstart through the real loopback SDK server."""
from pathlib import Path
import re


def test_readme_local_container_smoke(capsys):
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    section = readme.split("## Runnable local example (no provider credentials)", 1)[1]
    source = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    exec(compile(source, "README.md", "exec"), {"__name__": "__main__"})
    assert "reward: 1.0" in capsys.readouterr().out
