"""
System to store large data files on a server
Load them whem required by the user
Update the pointer file on github when new datafiles become available

Pro: Versioning is effectively done by Git
Con: Need to run server
"""

import hashlib
import json
import logging
import os
import shutil
import warnings
from pathlib import Path

import requests
import wget
from tqdm import tqdm

from .config import Config

logger = logging.getLogger(__name__)

# We are lazy and want a simple check if a file is in the Path
Path.__contains__ = lambda self, key: (self / key).exists()


class LargeFileStorage:
    """
    Download large data files from data server when needed
    New versions of the datafiles are indicated in a 'pointer' file
    that includes the hash of the newest version of the files

    Raises
    ------
    FileNotFoundError
        If the datafiles can't be located anywhere
    """

    def __init__(self, server, pointers, storage, cache):
        #:Server: Large File Storage Server address
        self.server = Server(server)

        if isinstance(pointers, str):
            path = Path(__file__).parent / pointers
            pointers = LargeFileStorage.load_pointers_file(path)

        #:dict(fname:hash): points from a filename to the current newest object id, usually a hash
        self.pointers = pointers
        #:Directory: directory of the current data files
        self.current = Path(storage).expanduser().absolute()
        #:Directory: directory for the cache
        self.cache = Path(cache).expanduser().absolute()
        #:dict(fname:hash): hashes of existing files, to avoid recalculation
        self._hashes = {}

    @staticmethod
    def load_pointers_file(filename):
        try:
            with open(str(filename), "r") as f:
                pointers = json.load(f)
        except FileNotFoundError:
            logger.error("Could not find LargeFileStorage reference file %s", filename)
            pointers = {}
        return pointers

    def hash(self, filename, blocks_per_iter=256, max_blocks=1000):
        """Hash a file, so we can compare with the server

        Parameters
        ----------
        filename : str
            filename
        blocks_per_iter : int, optional
            Number of blocks to hash per iteration, by default 256
        max_blocks : int, optional
            Maximum number of blocks to hash, by default 1000.
            A smaller number will limit the execution time. The current
            default should keep it to less than a second.

        Returns
        -------
        hash : str
            hexadecimal representation of the hash
        """
        hasher = hashlib.blake2b()
        blocksize = blocks_per_iter * hasher.block_size
        with open(str(filename), "rb") as f:
            for chunk, _ in zip(
                iter(lambda: f.read(blocksize), b""), range(max_blocks)
            ):
                hasher.update(chunk)
        return hasher.hexdigest()

    def symlink(self, src, dest):
        try:
            try:
                os.remove(dest)
            except:
                pass
            os.symlink(src, dest)
        except OSError as ex:
            # Might Fail on Windows, then just copy the file
            logger.debug(ex)
            warnings.warn(
                "Could not create symlinks, see https://docs.python.org/3/library/os.html#os.symlink for more details",
            )
            shutil.copy(src, dest)

    def get(self, key):
        """
        Request a datafile from the LargeFileStorage
        Assures that tracked files are at the specified version
        And downloads data from the server if necessary

        Parameters
        ----------
        key : str
            Name of the requested datafile

        Raises
        ------
        FileNotFoundError
            If the requested datafile can not be found anywhere

        Returns
        -------
        fullpath : str
            Absolute path to the datafile
        """
        key = str(key)

        # Step 1: Check if the file is tracked and/or exists in the storage directory
        if key not in self.pointers:
            if key not in self.current:
                if not os.path.exists(key):
                    raise FileNotFoundError(
                        f"File {key} does not exist and is not tracked by the Large File system"
                    )
                else:
                    logger.warning(
                        f"Data file {key} exists, but is not tracked by the large file storage"
                    )
                    return str(key)
            else:
                logger.warning(
                    f"Data file {key} exists, but is not tracked by the large file storage"
                )
                return str(self.current / key)

        # Step 2: Check Pointer version, i.e. newest version
        newest = self.pointers[key]

        if key in self.current:
            # Step 3: If newest version == storage version, we are all good and can use it
            if key in self._hashes.keys():
                current_hash = self._hashes[key]
            else:
                current_hash = self.hash(self.current / key)
                self._hashes[key] = current_hash
            if current_hash == newest:
                return str(self.current / key)

        # Step 4: Otherwise check the cache for the requested version
        if newest in self.cache:
            logger.debug("Using cached version of datafile")
            self.symlink(str(self.cache / newest), str(self.current / key))
            return str(self.current / key)

        # Step 5: If not in the cache, download from the server
        logger.info("Downloading newest version of %s from server", key)
        try:
            self.server.download(newest, self.cache)
            self.symlink(str(self.cache / newest), str(self.current / key))
        except TimeoutError:
            logger.warning("Server connection timed out.")
            if key in self.current:
                logger.warning("Using obsolete, but existing version")
            else:
                logger.warning("No data available for use")
                raise FileNotFoundError("No data could be found for the requested file")

        return str(self.current / key)

    def clean_cache(self):
        """ Remove unused cache files (from old versions) """
        used_files = self.pointers.values()
        for f in self.cache.iterdir():
            if f not in used_files:
                os.remove(f)

    def delete_file(self, fname):
        """ Delete a file, including the cache file """
        # Delete the file
        try:
            os.remove(str(self.current / fname))
        except OSError:
            pass

        # Delete the associated cache file
        try:
            p = self.pointers[fname]
            os.remove(str(self.cache / p))
        except (KeyError, IOError):
            pass

    def generate_pointers(self):
        """ Generate the pointers dictionary from the existing storage directory """
        pointers = {}
        for path in self.current.iterdir():
            name = path.name
            if not path.is_dir():
                pointers[name] = self.hash(path)

        # Only update existing files, keep old references
        self.pointers.update(pointers)
        return self.pointers

    def move_to_cache(self):
        """ Move currently used files into cache directory and use symlinks instead,
        just as if downloaded from a server """
        for fullpath in self.current.iterdir():
            name = fullpath.name
            if fullpath.is_file():
                # Copy file
                shutil.copy(str(fullpath), str(self.cache / self.pointers[name]))
                os.remove(str(fullpath))
                self.symlink(
                    str(self.cache / self.pointers[name]), str(self.current / name)
                )

    def create_pointer_file(self, filename):
        """ Create/Update the pointer file with new hashes """
        if self.pointers is None:
            raise RuntimeError("Needs pointers")

        with open(filename, "w") as f:
            json.dump(self.pointers, f, indent=4)


class Server:
    def __init__(self, url):
        self.url = url

    def download(self, fname, location):
        url = self.url + "/" + fname
        loc = str(location)
        os.makedirs(loc, exist_ok=True)
        wget.download(url, out=loc)
        print("\n")

    def isUp(self):
        try:
            r = requests.head(self.url)
            return r.status_code == 200
        except:
            return False
        return False


def setup_atmo(config=None):
    if config is None:
        config = Config()
    server = config["data.file_server"]
    storage = config["data.atmospheres"]
    cache = config["data.cache.atmospheres"]
    pointers = config["data.pointers.atmospheres"]
    lfs_atmo = LargeFileStorage(server, pointers, storage, cache)
    return lfs_atmo


def setup_nlte(config=None):
    if config is None:
        config = Config()
    server = config["data.file_server"]
    storage = config["data.nlte_grids"]
    cache = config["data.cache.nlte_grids"]
    pointers = config["data.pointers.nlte_grids"]
    lfs_nlte = LargeFileStorage(server, pointers, storage, cache)
    return lfs_nlte


def setup_lfs(config=None, lfs_atmo=None, lfs_nlte=None):
    if config is None:
        config = Config()
    if lfs_atmo is None:
        lfs_atmo = setup_atmo(config)
    if lfs_nlte is None:
        lfs_nlte = setup_nlte(config)
    return config, lfs_atmo, lfs_nlte


def get_available_atmospheres(config=None):
    if config is None:
        config = Config()
    pointers = config["data.pointers.atmospheres"]
    storage = config["data.atmospheres"]
    data = get_available_files(pointers, storage)
    return data


def get_available_nlte_grids(config=None):
    if config is None:
        config = Config()
    pointers = config["data.pointers.nlte_grids"]
    storage = config["data.nlte_grids"]
    data = get_available_files(pointers, storage)
    return data


def get_available_files(pointers, storage):
    pointers = Path(__file__).parent / pointers
    storage = Path(storage).expanduser()
    data = LargeFileStorage.load_pointers_file(pointers)
    files = list(data.keys())
    files_non_lfs = [
        f
        for f in os.listdir(storage)
        if f not in data and not os.path.isdir(storage / f)
    ]
    files += files_non_lfs
    return files

