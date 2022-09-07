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

# دى عشان لو مكتبات معتمدة عل حاجات تانية عندى ف المكتبة بينزلهالك تلقائى 
# with open('requirements.txt','r') as fr:
#     requires = fr.read().split('\n')
setuptools.setup(
    name = 'binary_brains' ,
    version = '0.0.3' , 
    author = 'Zakaria Ahmed' ,
    author_email='zakaria.ahmed6765@gmail.com',
    description= 'A small example package' , 
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

# -----------------------------------------------------------------------------------------
# setuptools.setup(
#     # pip install binary_brains
#     name = 'binary_brains' , # Replace with your own username
#     version = '0.0.1' , 
#     author = 'Zakaria Ahmed' ,
#     author_email='zakaria.ahmed6765@gmail.com',
#     description= 'A small example package' , 
#     long_description= long_description ,
#     long_description_content_type = 'text/markdown' ,
#     url= '' , # لو انت عامل بروجيكت وضيفت فيه المكتبة بتاعتك ضيف عنوانه لما انت رفعته عل github 
#     project_urls ={
#         'Bug Tracker' : 'https://github.com/pypa/sampleproject/issues' ,
#     },
#     classifiers= classifiers
#     # هنا بتعرفه المسار اللى المكتبة بتاعتك فيه 
#     package_dir={"":"src"} ,
#     packages=setuptools.find_packages(where="src") , 
#     python_requires=">=3.6" ,
#     #install_requires = requires ,
# )