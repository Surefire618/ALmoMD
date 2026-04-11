from setuptools import setup, find_packages

setup(
    name='ALmoMD',
    version='0.2.0',
    author='Kisung Kang',
    description='Active-learning machine-operated molecular dynamics (ALmoMD) is a Python code package designed for the effective training of machine learned interatomic potential (MLIP) through active learning based on uncertainty evaluation. It also facilitates the implementation of molecular dynamics (MD) using trained MLIPs with uncertainty evaluation.',
    packages=find_packages(),
    install_requires=[
        'nequip >= 0.5.6, < 0.7',
        'fhi-vibes',
        'numpy',
        'scipy',
        'ase',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'almomd = almd.cli:main'
        ]
    },
)

