import subprocess
import sys
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        """Compile C++ shared library during build."""
        sys.stdout.write("=" * 60 + "\n")
        sys.stdout.write("Building C++ shared library...\n")
        sys.stdout.write("=" * 60 + "\n")
        sys.stdout.flush()
        
        # Path to the compiled directory
        compiled_dir = Path(self.root) / "src" / "tmg_hmc" / "compiled"
        
        if not compiled_dir.exists():
            raise RuntimeError(f"Compiled directory not found: {compiled_dir}")
        
        # Run make in the compiled directory
        try:
            result = subprocess.run(
                ["make"],
                cwd=compiled_dir,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            sys.stdout.write("=" * 60 + "\n")
            sys.stdout.write("C++ library compiled successfully!\n")
            sys.stdout.write("=" * 60 + "\n")
            sys.stdout.flush()
        except subprocess.CalledProcessError as e:
            sys.stderr.write(f"Make failed!\n")
            raise RuntimeError("Failed to compile C++ library") from e
        except FileNotFoundError:
            raise RuntimeError("Make command not found. Please install build-essential.")