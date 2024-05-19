import os
import glob
import numpy as np
from tqdm.auto import tqdm

def folder_exists(folderpath):
    """Check whether a folder exists by given folder path."""
    return os.path.isdir(folderpath)

def del_file(filepath):
    """Delete a file by given file path."""
    os.remove(filepath)

def exists_or_mkdir(dir):
    if not folder_exists(dir):
        os.makedirs(dir)

def load_folder_list(path=""):
    """Return a folder list in a folder by given a folder path.

    Parameters
    ----------
    path : str
        A folder path.

    """
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

def get_image_fns_from_dir(directory):
    """
    Returns array of paths to all jpg and png files in a given directory.
    Doesn't check subfolders.
    """
    png_files = np.array([file for file in glob.glob(directory + "*.png")])
    jpg_files = np.array([file for file in glob.glob(directory + "*.jpg")])
    return np.concatenate([png_files,jpg_files],axis=0)

def get_filetype_fns_from_dir(directory, extension='.json'):
    """
    Returns array of paths to all "extension" type files in a given directory.
    Doesn't check subfolders.
    """
    _dir = directory if directory.endswith('/') else directory + '/'
    return np.array([file for file in glob.glob(_dir + f'*{extension}')])

def get_image_fns_from_list_dirs(dirs):
    """
    Returns list of paths to all jpg and png type files in a given directories.
    Doesn't check subfolders.
    """
    existing_dirs = [_dir for _dir in dirs if folder_exists(_dir)]
    if len(existing_dirs) == 0:
        return []
    all_image_paths = [get_image_fns_from_dir(_dir) for _dir in existing_dirs]
    all_image_paths = [item for sublist in all_image_paths for item in sublist]
    return all_image_paths

def get_filetype_fns_from_list_dirs(dirs, extension='.json'):
    """
    Returns list of paths to all extension type files in a given directories.
    Doesn't check subfolders.
    """
    existing_dirs = [_dir for _dir in dirs if folder_exists(_dir)]
    if len(existing_dirs) == 0:
        return []
    all_paths = [get_filetype_fns_from_dir(_dir, extension=extension) for _dir in existing_dirs]
    all_paths = [item for sublist in all_paths for item in sublist]
    return all_paths

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

def find_basename_matches(key_files, value_files):
    """ Match lists of two files to dictionary map by their basenames """
    basename_key_fns = [os.path.basename(fn).split('.')[0] for fn in key_files]
    basename_value_fns = [os.path.basename(fn).split('.')[0] for fn in value_files]
    key_val_map = {}
    for i, key_fn in tqdm(enumerate(basename_key_fns), 'matching file pairs'):
        condition = lambda x: x.startswith(key_fn) 
        value_indices = find_indices(basename_value_fns, condition)
        if len(value_indices) > 0:
            key_val_map[key_files[i]] = value_files[value_indices[0]]
    return key_val_map

def get_last_directory(root = './output/'):
    """ Returns alphabetically last directory name """
    dirs = [os.path.join(root, _dir) for _dir in os.listdir(root) if os.path.isdir(os.path.join(root, _dir))]
    dirs.sort()
    return dirs[-1] # latest

def get_matching_file(dir, startswith=None, endswith=None):
    assert not(startswith is None and endswith is None)
    all_files = os.listdir(dir)
    if startswith is not None:
        fs = [f for f in all_files if f.startswith(startswith)]
    else:
        fs = [f for f in all_files if f.endswith(endswith)]
    if len(fs) == 0:
        return None
    return fs[-1]
