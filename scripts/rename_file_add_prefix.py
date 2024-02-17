import os


def add_prefix_to_files(path, prefix):
    """Add a prefix to all files in the specified directory."""
    if not os.path.exists(path):
        print(f"The path '{path}' does not exist.")
        return

    # List all items in the directory
    items = os.listdir(path)

    # Filter out directories, only process files
    files = [item for item in items if os.path.isfile(
        os.path.join(path, item))]

    for filename in files:
        new_name = prefix + filename
        old_path = os.path.join(path, filename)
        new_path = os.path.join(path, new_name)

        # Rename the file with the prefix
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_name}")


if __name__ == "__main__":
    path = input("Enter the path of the directory: ").strip()
    prefix = input("Enter the prefix to add to files: ").strip()

    add_prefix_to_files(path, prefix)
