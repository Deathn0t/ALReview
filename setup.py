from setuptools import setup, find_packages

print(find_packages(where="src"))

setup(
    name="ALreview",  # active learning review
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["modAL", "pandas", "numpy", "scikit-learn", "rich"],
    entry_points={  # Optional
        "console_scripts": [
            "alreview=ALreview.run:main",
        ],
    },
    # package data
    include_package_data=True,
    package_data={"": ["ALreview/console.theme"]},
)
