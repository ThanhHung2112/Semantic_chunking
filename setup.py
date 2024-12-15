from setuptools import setup, find_packages

setup(
    name='semantic_chunking',
    version='0.1.1',
    description='A Python library for semantic text chunking using Sentence Transformers.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Le Thanh Hung',
    author_email='lthung2112@gmail.com',
    url='https://github.com/thanhhung2112/semantic_chunking',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.5',
        'sentence-transformers>=2.2.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
