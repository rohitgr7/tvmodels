from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

exec(open('tvmodels/version.py').read())

setup(
    name='tvmodels',
    version=__version__,
    description='Implementation of vision models with their pretrained weights',
    packages = find_packages(),
    include_package_data=True,

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
