from setuptools import setup

with open('README.md') as f:
    readme = f.read()


setup(
    name='tvmodels',
    version='0.0.1',
    description='Implementation of vision models with theri pretrained weights',
    py_modules=['tvmodels'],
    package_dir={'': 'tvmodels'},

    long_description=readme,
    long_description_content_type='text/markdown',

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
