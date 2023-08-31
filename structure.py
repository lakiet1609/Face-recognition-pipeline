from pathlib import Path
import os

list_files = [
    'src/components/__init__.py',
    'src/utils/__init__.py',
    'src/components/__init__.py',
    'src/models/__init__.py',
    'src/config/__init__.py',
    'src/database/__init__.py',
    'src/api/__init__.py',
    'configs/.gitkeep',
    'models/.gitkeep',
    'main.py',
    'requirements.txt'
]

for file in list_files:
    file = Path(file)
    file_dir, file_name = os.path.split(file)

    if file_dir != '':
        os.makedirs(file_dir, exist_ok=True)
    
    if (not os.path.exists(file)) or (os.path.getsize(file) == 0):
        with open(file, 'w') as f:
            pass