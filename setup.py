import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='py_dof',  
     version='0.0',
     package_dir={'py_dof': 'py_dof'},
     package=['py_dof','py_dof/test'], 
     author="R.LAPLAZA",
     author_email="laplazasolanas@gmail.com",
     description="Calculate DOFs using PySCF.",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/rlaplaza/py_dof",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
     ],
 )
