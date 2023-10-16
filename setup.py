from setuptools import setup


import post_crisis_monitoring


def get_requirements(requirements_path: str = 'requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]


setup(
    name='post_crisis_monitoring',
    version=post_crisis_monitoring.__version__,
    description='Post Crisis Monitoring',
    author='Roger Garriga',
    packages=['post_crisis_monitoring'],
    install_requires=get_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3.9.8'
    ],
)
