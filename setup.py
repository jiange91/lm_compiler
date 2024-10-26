from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='compiler',
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'cognify=compiler.__main__:main',  # Assuming you have a `main()` function in __main__.py
        ],
    },
)