def extend_path(path: str, overwrite: bool = False) -> str:
    if overwrite:
        return path
    else:
        (root, ext) = os.path.splitext(path)
        i = 0
        while os.path.exists(root + str(i) + ext):
            i += 1
        return (root + str(i) + ext)
