"""全局配置单例"""

from app.settings.loader import load_config

# 延迟加载的配置单例
_config = None


def get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config():
    global _config
    _config = None
