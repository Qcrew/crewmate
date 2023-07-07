from setuptools import setup

setup(
    name='crewmate',
    version='0.0.0.6',
    url="https://github.com/Qcrew/crewmate",
    author="Qcrew",
    author_email="general.qcrew@gmail.com",
    description='',
    py_modules=['crewmate.utils', 'crewmate.qctrl',
                'crewmate.fit', 'crewmate.cost_functions'],
    package_dir={
        '': 'src',
    },
    install_requires=[
        "numpy"
    ],
    extras_require={
        "dev": [
            "pytest>=3.7",
            "check-manifest>=0.49"
        ]
    }
)
