import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='LGBtrainer',  

     version='0.1',

     py_modules=['LGBtrainer'],
     
     package_dir={'': 'src'},	

     author="Rajwardhan Shinde",

     author_email="rajshinde55553@gmail.com",

     description="Find LGBM Hyperparams and train the model",
     long_description=open("README.md").read(),
     long_description_content_type="text/markdown",
     url="https://github.com/Rajshinde07/LGBtrainer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
	"Programming Language :: Python :: 3.6",
	"Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
