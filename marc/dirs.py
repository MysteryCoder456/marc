from platformdirs import PlatformDirs

_DIRS = PlatformDirs("Marc", "CodeBoi", ensure_exists=True)

CHATS_PATH = _DIRS.user_data_path / "chats"


def ensure_paths():
    CHATS_PATH.mkdir(exist_ok=True)
