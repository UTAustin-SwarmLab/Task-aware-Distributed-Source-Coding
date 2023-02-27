from setuptools import setup

setup(name='dtac',
      version='1.0.0',
      description='Distributed Task-Aware Compression',
      packages=['dtac'],
      install_requires=['gym==0.21.0',
                        'mujoco-py',
                        'matplotlib',
                        'scikit-image',
                        'tensorboard',
                        'torch',
                        'torchvision',
                        'torchaudio'])