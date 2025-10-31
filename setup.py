from setuptools import setup
from setuptools.command.build_py import build_py
import subprocess
import sys
import shutil
import platform
from pathlib import Path

class BuildPyWithMake(build_py):
    """Custom build_py that compiles C++ library first."""
    
    def run(self):
        # Build the C++ library
        self.build_cpp_library()
        # Run normal build_py
        super().run()
    
    def build_cpp_library(self):
        """Compile the C++ shared library using make."""
        print("=" * 70)
        print("Building C++ shared library with makefile")
        print("=" * 70)
        
        # Source directory
        source_dir = Path('src/tmg_hmc/compiled')
        if not source_dir.exists():
            raise RuntimeError(f"Source directory not found: {source_dir}")
        
        # Get make command
        make_cmd = 'make'
        if platform.system() == 'Windows':
            make_cmd = shutil.which('mingw32-make') or shutil.which('make') or 'make'
        
        # Run make
        try:
            subprocess.run(
                [make_cmd, 'all'],
                cwd=source_dir,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            print("=" * 70)
            print("C++ library built successfully")
            print("=" * 70)
        except Exception as e:
            raise RuntimeError(f"Failed to build C++ library: {e}")

setup(
    cmdclass={'build_py': BuildPyWithMake},
    package_data={
        'tmg_hmc': ['compiled/*.so', 'compiled/*.dylib', 'compiled/*.dll'],
    },
    include_package_data=True,
)