__all__ = [
    "set_cache_dir",
    "set_data_dir",
    "default_cache_dir",
    "default_data_dir",
    "list_cache",
    "list_data",
    "delete_from_cache",
    "delete_from_data",
    "_storage_factory",
    "dataset_path",
    "model_path",
    "encoder_path",
]

import os

import synstore

from abstract2gene import __name__ as pkg_name

synstore.set_package_name(pkg_name)

set_cache_dir = synstore.set_cache_dir
set_data_dir = synstore.set_data_dir
default_cache_dir = synstore.default_cache_dir
default_data_dir = synstore.default_data_dir
list_cache = synstore.list_cache
list_data = synstore.list_data
delete_from_cache = synstore.delete_from_cache
delete_from_data = synstore.delete_from_data
_storage_factory = synstore.storage_factory

def dataset_path(name: str | None) -> str:
    """Return a path below the default datasets path.

    The default path for storing datasets is a function of `default_data_dir`,
    setting this will change the results of `dataset_path`.
    """
    path = default_data_dir("datasets")
    if name:
        return os.path.join(path, name)
    return path

def model_path(name: str | None) -> str:
    """Return a path below the default models path.

    The default path for storing models is a function of `default_data_dir`,
    setting this will change the results of `dataset_path`.
    """
    path = default_data_dir("models")
    if name:
        return os.path.join(path, name)
    return path

def encoder_path(name: str | None) -> str:
    """Return a path below the default encoders path.

    The default path for storing encoders is a function of `default_data_dir`,
    setting this will change the results of `dataset_path`.
    """
    path = default_data_dir("encoders")
    if name:
        return os.path.join(path, name)
    return path
