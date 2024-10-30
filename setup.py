from setuptools import setup, find_packages

setup(
    name='DNLP-Simulation-Service',
    version='0.1.0',  
    author="Mohamed Hamdey",
    author_email="mohamed.hamdey@gmail.com",
    description='A deep learning natural language processing service for simulating different systems',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown', 
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'transformers',
        'wandb',
        'scikit-learn',
        'numpy',
    ],
)
