# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:23:07 2024

@author: dforc
"""

import pkg_resources

required_packages = ['asyncpraw', 'pandas', 'aiohttp', 'python-dotenv', 'nest_asyncio', 'tqdm']
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

for package in required_packages:
    if package in installed_packages:
        print(f"{package}=={installed_packages[package]}")