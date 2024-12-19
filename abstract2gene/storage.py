__all__ = [
    "default_cache_dir",
    "default_data_dir",
    "list_cache",
    "list_data",
    "delete_from_cache",
    "delete_from_data",
    "_storage_factory",
]

import synstore

from abstract2gene import __name__ as pkg_name

synstore.set_package_name(pkg_name)

default_cache_dir = synstore.default_cache_dir
default_data_dir = synstore.default_data_dir
list_cache = synstore.list_cache
list_data = synstore.list_data
delete_from_cache = synstore.delete_from_cache
delete_from_data = synstore.delete_from_data
_storage_factory = synstore.storage_factory
