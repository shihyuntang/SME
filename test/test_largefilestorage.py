import pytest

from os import listdir
from os.path import dirname, exists
import requests

from pysme.large_file_storage import LargeFileStorage, setup_atmo, setup_nlte
from pysme.config import Config


def lfs_available():
    config = Config()
    r = requests.head(config["data.file_server"])
    return r.status_code == 200


skipif_lfs = pytest.mark.skipif(lfs_available(), reason="LFS not available")


@pytest.fixture
def lfs_nlte():
    lfs_nlte = setup_nlte()
    yield lfs_nlte


@pytest.fixture
def lfs_atmo():
    lfs_atmo = setup_atmo()
    yield lfs_atmo
