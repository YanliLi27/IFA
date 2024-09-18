import os 
try:
    from setuptools import setup, Extension
except:
    from distutils.core import setup, Extension

ex = []

for i, j, k in os.walk('./'):
    for file in k:
        if file.endswith('.pyx'):
            ex.append(Extension('*', [(i if i.endswith('/') else i + '/') + file]))

this_directory = os.path.abspath(os.path.dirname(os.path.realpath('__file__')))

def read_file(filename):
    with open(os.path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]

setup(name='IFA',
    python_requires='>=3.6.0',
    version='0.1',
    author='Yanli Li',
    author_email='yanlili.cn@gmail.com',
    url='https://github.com/YanliLi27/IFA',
    description='A tool for calibrating Class Activation Maps',
    packages=['cam_components', '.'],
    include_package_data=True,
    keywords=['CAM', 'DL', 'AI'],
    install_requires=read_requirements('requirements.txt'),
    package_data={
        "cam_components": ["*.pyi", "**/*.pyi", "*.py", "*/*.py", "*.pxd", "*/*.pxd", '*/*.pyd', '*.pyd', '*.pyx', '*/*.pyx'],
        '.': ['requirements.txt']
    },
    license="MIT"
    )

# python ifasetup.py check
# python ifasetup.py sdist
