from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
from pathlib import Path

class MakeBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name == "tmg_hmc.compiled._ext":
            self.compile_with_make()
        else:
            super().build_extension(ext)
    
    def compile_with_make(self):
        print("=" * 60)
        print("Building C++ shared library with make...")
        print("=" * 60)
        
        compiled_dir = Path(self.build_lib) / "tmg_hmc" / "compiled"
        src_compiled_dir = Path("src") / "tmg_hmc" / "compiled"
        
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
            print("C++ library compiled successfully!")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Failed to compile C++ library") from e

# Define extension that triggers make
ext = Extension(
    name="tmg_hmc.compiled._ext",
    sources=["src/tmg_hmc/compiled/utils.cpp"],
    extra_compile_args=["-O3", "-Wall", "-fPIC", "-std=c++14"],
    language="c++",
)

setup(
    ext_modules=[ext],
    cmdclass={'build_ext': MakeBuildExt},
    package_data={
        'tmg_hmc': ['compiled/*.so', 'compiled/*.dylib', 'compiled/*.dll'],
    },
)