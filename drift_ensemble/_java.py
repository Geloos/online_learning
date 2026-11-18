"""JAVA_HOME bootstrap.

CapyMOA runs MOA through a JVM (via jpype) and needs JAVA_HOME. On this
machine java is on PATH but JAVA_HOME is not set, so we derive it from the
JVM's own ``java.home`` property before any capymoa import happens.

Import order contract: every module that (directly or indirectly) imports
capymoa must import ``drift_ensemble.config`` first, which calls
``ensure_java_home()`` at import time.
"""

import os
import subprocess


def ensure_java_home() -> str:
    """Set os.environ['JAVA_HOME'] from the running JVM if it is unset."""
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        return java_home
    try:
        proc = subprocess.run(
            ["java", "-XshowSettings:properties", "-version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "JAVA_HOME is not set and 'java' was not found on PATH. "
            "Install a JDK 11+ or set JAVA_HOME."
        ) from exc
    # -XshowSettings prints the properties on stderr.
    for line in proc.stderr.splitlines():
        line = line.strip()
        if line.startswith("java.home"):
            java_home = line.split("=", 1)[1].strip()
            os.environ["JAVA_HOME"] = java_home
            return java_home
    raise RuntimeError(
        "Could not derive JAVA_HOME from 'java -XshowSettings:properties'. "
        "Set JAVA_HOME manually."
    )
