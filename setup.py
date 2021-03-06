from setuptools import setup

setup(
    name="sivio",
    version="0.1.0",
    description="Python software for radio ionospheric simulations",
    url="http://github.com/kariukic/sivio",
    author="Kariuki Chege",
    author_email="jameskariuki31@gmail.com",
    license="MIT",
    packages=["sivio"],
    keywords="Radio telescope, Ionosphere",
    zip_safe=False,
    scripts=["scripts/sivio_wrapper.py"],
    install_requires=[
        "numpy",
        "pandas",
        "numba",
        "scipy",
        "astropy",
        "matplotlib",
        "future",
        "astropy",
        "astroquery",
        "psutil",
        "python-casacore",
        "mwa_pb",
        "cthulhu",
        "AegeanTools",
    ],
)
