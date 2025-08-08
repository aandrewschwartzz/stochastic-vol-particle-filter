# from setuptools import setup, Extension
# from Cython.Build import cythonize
# import numpy as np
# import os

# # Path to the .pyx file in the same directory as setup.py
# base_dir = os.path.dirname(os.path.abspath(__file__))
# pyx_path = os.path.join(base_dir, 'trajectory_list.pyx')  # No subdirectories needed

# # Define the extension module including the path to NumPy headers
# extensions = [
#     Extension("trajectory_list", [pyx_path],
#               include_dirs=[np.get_include()]),  # Adds the NumPy headers to the path
# ]

# setup(
#     ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}, verbose=True),
#     script_args=['build_ext', '--inplace']
# )

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import scipy
import os

# Path to the .pyx file in the same directory as setup.py
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'ABC_CAPF.pyx')  # Adjust the filename as necessary
scipy_include = os.path.join(os.path.dirname(scipy.__file__), 'special')

# Define the extension module including the path to NumPy headers
extensions = [
    Extension("ABC_CAPF", [file_path],
              include_dirs=[np.get_include()]),  # Adds the NumPy headers to the path
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}, verbose=True),
    script_args=['build_ext', '--inplace']
)