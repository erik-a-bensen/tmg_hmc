# clean.py
import shutil
from pathlib import Path

dirs_to_remove = ['build', 'dist', '.eggs']
patterns_to_remove = ['*.egg-info', '__pycache__', '*.pyc', '*.pyo']

for dir_name in dirs_to_remove:
    if Path(dir_name).exists():
        shutil.rmtree(dir_name)

for pattern in ['*.egg-info', '__pycache__']:
    for path in Path('.').rglob(pattern):
        if path.is_dir():
            shutil.rmtree(path)

# Remove compilation artifacts
for pattern in ['*.so', '*.pyd', '*.dll']:
    for path in Path('.').rglob(pattern):
        if path.is_file():
            path.unlink()

# Remove build log file 
build_log = Path('build.log')
if build_log.exists():
    build_log.unlink()