"""Put the image's own package on the path.

The image dir is not installed; the Dockerfile copies it to /app. Tests run
against the source tree the same way the container runs the copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
