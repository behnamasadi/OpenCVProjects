import os
import re


def rename_files_with_prefix(directory_path, prefix):
    # Get all files in the directory
    files = os.listdir(directory_path)

    # Loop through each file
    for file in files:
        # If the file matches the format <number>.jpg
        if re.match(r'^\d+\.jpg$', file):
            # Create a new file name with the given prefix
            new_file_name = f"{prefix}{file}"
            # Rename the file
            os.rename(os.path.join(directory_path, file),
                      os.path.join(directory_path, new_file_name))


if __name__ == "__main__":
    dir_path = input("Enter the directory path: ")
    prefix = input("Enter the prefix to add: ")
    rename_files_with_prefix(dir_path, prefix)
    print("Files renamed successfully!")
