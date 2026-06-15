from platformdirs import PlatformDirs

DIRS = PlatformDirs("Marc", "CodeBoi", ensure_exists=True)
DIRS.user_data_path.mkdir(exist_ok=True)
