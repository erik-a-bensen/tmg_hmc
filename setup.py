from setuptools import setup
from setuptools.command.build_py import build_py
import subprocess
import sys
from pathlib import Path

class BuildWithMake(build_py):
    """Custom build command that runs make before building."""
    
    def run(self):
        # Compile the C++ extension
        self.compile_extension()
        # Continue with normal build
        super().run()
    
    def compile_extension(self):
        """Run make to compile the C++ shared library."""
        print("=" * 60)
        print("Building C++ shared library...")
        print("=" * 60)
        sys.stdout.flush()
        
        # Path to the compiled directory
        compiled_dir = Path(__file__).parent / "src" / "tmg_hmc" / "compiled"
        
        if not compiled_dir.exists():
            raise RuntimeError(f"Compiled directory not found: {compiled_dir}")
        
        # Run make
        try:
            result = subprocess.run(
                ["make"],
                cwd=compiled_dir,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            print("=" * 60)
            print("C++ library compiled successfully!")
            print("=" * 60)
            sys.stdout.flush()
        except subprocess.CalledProcessError as e:
            sys.stderr.write("Make failed!\n")
            raise RuntimeError("Failed to compile C++ library") from e
        except FileNotFoundError:
            raise RuntimeError(
                "Make command not found. Please install build tools:\n"
                "  Linux: sudo apt-get install build-essential\n"
                "  macOS: xcode-select --install\n"
                "  Windows: Install Visual Studio Build Tools"
            )

setup(
    cmdclass={
        'build_py': BuildWithMake,
    },
    package_data={
        'tmg_hmc': ['compiled/*.so', 'compiled/*.dylib', 'compiled/*.dll'],
    },
    include_package_data=True,
)