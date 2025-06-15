"""Setup configuration for agent-interrogator."""

from setuptools import setup, find_packages

setup(
    name="agent-interrogator",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "rich>=14.0.0",
        "pydantic>=2.0.0",
        "pytest>=8.0.0",
        "pytest-asyncio>=1.0.0"
    ],
    python_requires=">=3.9",
)
