from setuptools import setup

with open('README.md') as f:
    readme = f.read()


setup(
    name='tvmodels',
    version='0.0.6',
    description='Implementation of vision models with their pretrained weights',
    py_modules=['tvmodels'],
    package_dir={'': 'tvmodels'},

    url='https://github.com/rohitgr7/tvmodels',
    author='Rohit Gupta',
    author_email='rohitgr1998@gmail.com',

    long_description=readme,
    long_description_content_type='text/markdown',

    install_requires=[
        'torch>=1.0.1',
        'numpy',
        'requests'],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent'
    ]
)
