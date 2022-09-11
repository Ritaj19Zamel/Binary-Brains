import setuptools 

classifiers=[
        'Development Status :: 5 - Production/Stable' ,
	    'Intended Audience :: Education' , 
	    'Operating System :: Microsoft :: Windows :: Windows 10' , 
	    'License :: OSI Approved :: MIT License' ,
	    'Programming Language :: Python :: 3' 
    ]
with open('README.md' , 'r' , encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'Package_for_math' ,
    version = '0.0.1' , 
    author = 'Ahmed Ibrahim' ,
    author_email='ahmed.ibrahim797379@gmail.com',
    description= 'A small example package for mathematical operation' , 
    long_description= long_description ,
    long_description_content_type = 'text/markdown' ,
     url= '' ,
     project_urls ={
        'Bug Tracker' : 'https://github.com/pypa/sampleproject/issues' ,
    },
    classifiers= classifiers ,
    package_dir={"":"src"} ,
    packages=setuptools.find_packages(where="src") , 
    python_requires=">=3.6" ,
)
