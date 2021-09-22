import os
from os.path import join
from shutil import copyfile

folder = "/DATA/sme/nlte_grids"
content = os.listdir(folder)

for file in content:
    try:
        src = os.readlink(join(folder, file))
        dst = "%s_v1.0.0.%s" % (*file.rsplit(".", 1),)
        dst = join(folder, dst)
        copyfile(src, dst, follow_symlinks=True)
    except OSError as ex:
        # Not a symlink so everything is good
        pass
