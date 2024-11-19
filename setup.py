from setuptools import setup, find_packages
import os

readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, encoding='utf-8') as fh:
        long_description = fh.read()
else:
    long_description = "A deep learning natural language processing service for simulating different systems (normal-parallel-single server)."

setup(
    name='SimsysDNLP',
    version='0.0.4',  
    author="Mohamed Hamdey",
    author_email="mohamed.hamdey@gmail.com",
    description='A deep learning natural language processing service for simulating different systems (normal-parallel-single server).',
    long_description=long_description,
    long_description_content_type='text/markdown',  
    url="https://github.com/Mohamed-Hamdey/DNLP-Simulation-Service",  
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
