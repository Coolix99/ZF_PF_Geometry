from setuptools import setup, find_packages

setup(
    name="zf_pf_geometry",
    version="0.1.0",
    description="Analyses for the zebrafish pectoral fin geometry.",
    author="Maximilian Kotz",
    author_email="maximilian.kotz@tu-dresden.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "GitPython",
        "simple-file-checksum",
        "pyvista",
        "trimesh",
        "opencv-python",
        "pymeshfix",
        "pandas",
        "napari[all]>=0.5"
    ],
    python_requires='>=3.9',
)
