from setuptools import setup, Extension
import platform
import pybind11

extra_compile_args = ['-O3', '-std=c++14']
extra_link_args = []

if platform.system() == 'Linux':
    extra_compile_args.append('-fPIC')
    extra_link_args.extend(['-static-libstdc++', '-static-libgcc'])
elif platform.system() == 'Darwin':
    extra_compile_args.extend(['-fPIC', '-stdlib=libc++'])
    extra_link_args.append('-stdlib=libc++')
elif platform.system() == 'Windows':
    extra_compile_args = ['/O2', '/std:c++14']

setup(
    ext_modules=[
        Extension(
            'tmg_hmc.compiled',
            sources=['src/tmg_hmc/quad_solns.cpp'],
            include_dirs=[pybind11.get_include()],
            language='c++',
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
)
