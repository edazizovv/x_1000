#
import setuptools
from setuptools import setup


metadata = {'name': 'x_1000',
            'maintainer': 'Edward Azizov',
            'maintainer_email': 'edazizovv@gmail.com',
            'description': 'X_1000 Models Gen',
            'license': 'MIT',
            'url': 'https://github.com/edazizovv/x_1000',
            'download_url': 'https://github.com/edazizovv/x_1000',
            'packages': setuptools.find_packages(),
            'include_package_data': True,
            'version': '0.1.0',
            'long_description': '',
            'python_requires': '>=3.10',
            'install_requires': []}

setup(**metadata)
