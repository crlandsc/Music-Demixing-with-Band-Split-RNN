import os
import shutil
from tqdm import tqdm


def compare_and_copy_folders(source_dir, target_dir):
    """
    Compare the names of all folders within two separate directories of different names,
    and copy all files within the matching folders from the source directory to
    the target directory unless the file already exists in the target directory.

    Args:
        source_dir (str): Source directory path
        target_dir (str): Target directory path
    """
    # Get the list of directories in the source directory
    source_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    # Get the list of directories in the target directory
    target_dirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]

    # Compare and copy contents of matching folders
    for source_subdir in tqdm(source_dirs):
        if source_subdir in target_dirs:
            source_subdir_path = os.path.join(source_dir, source_subdir)
            target_subdir_path = os.path.join(target_dir, source_subdir)
            # print("Folders match: {} and {}".format(source_subdir_path, target_subdir_path))

            # Copy files from the source directory to the target directory
            for dirpath, dirnames, filenames in os.walk(source_subdir_path):
                for filename in filenames:
                    src_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(src_path, source_subdir_path)
                    dst_path = os.path.join(target_subdir_path, rel_path)
                    if not os.path.exists(dst_path):
                        # print("Copying {} to {}".format(src_path, dst_path))
                        shutil.copy2(src_path, dst_path)
                    # else:
                        # print("{} already exists in {}".format(filename, target_subdir_path))


source_dir = "I:/MDX-23/A_Label_Noise_norm"
target_dir = "I:/MDX-23/A_Label_Noise_Processed/Other/train"
compare_and_copy_folders(source_dir, target_dir)
