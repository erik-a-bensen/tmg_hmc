from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os
import shutil
import platform
from pathlib import Path

class MakefileExtension(Extension):
    """Custom extension that's built with a makefile."""
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class BuildMakefileExt(build_ext):
    """Build extension using makefile."""
    
    def run(self):
        """Build all extensions."""
        for ext in self.extensions:
            self.build_makefile_extension(ext)
    
    def build_makefile_extension(self, ext):
        """Build a single extension using its makefile."""
        print("=" * 70)
        print(f"Building {ext.name} with makefile")
        print("=" * 70)
        sys.stdout.flush()
        
        # Source directory with makefile
        source_dir = Path(ext.sourcedir)
        if not source_dir.exists():
            raise RuntimeError(f"Source directory not found: {source_dir}")
        
        # Build directory for output
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        extdir.mkdir(parents=True, exist_ok=True)
        
        # Determine make command
        make_cmd = self._get_make_command()
        
        # Set up environment
        env = os.environ.copy()
        
        # Platform-specific configuration
        if platform.system() == 'Darwin':
            # macOS - handle universal binaries if needed
            if 'ARCHFLAGS' in env:
                print(f"Building for architecture: {env['ARCHFLAGS']}")
        
        # Run make
        try:
            print(f"Running: {make_cmd} in {source_dir}")
            sys.stdout.flush()
            
            # Build
            subprocess.run(
                [make_cmd, 'all'],
                cwd=source_dir,
                env=env,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            
            print("=" * 70)
            print("Build successful!")
            print("=" * 70)
            sys.stdout.flush()
            
            # Copy built libraries to the package
            self._copy_libraries(source_dir, extdir)
            
        except subprocess.CalledProcessError as e:
            print("=" * 70)
            print(f"ERROR: Make failed with exit code {e.returncode}")
            print("=" * 70)
            raise RuntimeError(f"Failed to build {ext.name}") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Make command '{make_cmd}' not found. "
                f"Please install build tools for your platform."
            ) from e
    
    def _get_make_command(self):
        """Determine the appropriate make command for the platform."""
        if platform.system() == 'Windows':
            # Try to find mingw32-make, then make
            for cmd in ['mingw32-make', 'make']:
                if shutil.which(cmd):
                    return cmd
            raise RuntimeError(
                "No make command found. Please install MinGW or similar build tools."
            )
        else:
            # Unix-like systems
            return 'make'
    
    def _copy_libraries(self, source_dir, dest_dir):
        """Copy built libraries from source to destination."""
        # Platform-specific library extensions
        extensions = {
            'Linux': ['*.so'],
            'Darwin': ['*.dylib'],
            'Windows': ['*.dll'],
        }
        
        exts = extensions.get(platform.system(), ['*.so', '*.dylib', '*.dll'])
        
        copied = False
        for pattern in exts:
            for lib_file in source_dir.glob(pattern):
                dest_file = dest_dir / lib_file.name
                print(f"Copying {lib_file.name} to {dest_dir}")
                shutil.copy2(lib_file, dest_file)
                copied = True
        
        if not copied:
            print(f"WARNING: No libraries found in {source_dir}")

# Define the makefile-based extension
ext_modules = [
    MakefileExtension(
        'tmg_hmc.compiled',
        sourcedir='src/tmg_hmc/compiled'
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildMakefileExt},
    package_data={
        'tmg_hmc': ['compiled/*.so', 'compiled/*.dylib', 'compiled/*.dll'],
    },
    include_package_data=True,
)