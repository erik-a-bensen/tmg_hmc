from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os
import shutil
from pathlib import Path

class BuildExtWithMake(build_ext):
    """Custom build_ext command that runs make before building."""
    
    def run(self):
        # Compile the C++ extension first
        self.compile_extension()
        # Continue with normal build_ext
        super().run()
    
    def compile_extension(self):
        """Run make to compile the C++ shared library."""
        print("=" * 60)
        print("Building C++ shared library...")
        print("=" * 60)
        sys.stdout.flush()
        
        # Path to the compiled directory
        src_compiled_dir = Path(__file__).parent / "src" / "tmg_hmc" / "compiled"
        
        if not src_compiled_dir.exists():
            raise RuntimeError(f"Compiled directory not found: {src_compiled_dir}")
        
        # Run make
        try:
            subprocess.run(
                ["make", "clean"],
                cwd=src_compiled_dir,
                check=False,
            )
            subprocess.run(
                ["make"],
                cwd=src_compiled_dir,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            print("=" * 60)
            print("C++ library compiled successfully!")
            print("=" * 60)
            sys.stdout.flush()
            
            # Copy the compiled library to the build directory
            build_lib = Path(self.build_lib) / "tmg_hmc" / "compiled"
            build_lib.mkdir(parents=True, exist_ok=True)
            
            # Find and copy the compiled library
            for ext in ['*.so', '*.dylib', '*.dll']:
                for lib_file in src_compiled_dir.glob(ext):
                    dest = build_lib / lib_file.name
                    print(f"Copying {lib_file} to {dest}")
                    shutil.copy2(lib_file, dest)
                    
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

# Use actual source file for the extension to force platform-specific wheel
ext_modules = [
    Extension(
        name="tmg_hmc.compiled._ext",
        sources=["src/tmg_hmc/compiled/utils.cpp"],
        language="c++",
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtWithMake,
    },
    package_data={
        'tmg_hmc': ['compiled/*.so', 'compiled/*.dylib', 'compiled/*.dll'],
    },
    include_package_data=True,
)