"""Resolve trusted application paths before constructing a store or an agent."""
import json
from pathlib import Path


def resolve_dataset_path(config_path, *, override=None):
    """CLI override wins; configured dataset roots must be absolute and present."""
    if override is not None:
        root = Path(override).expanduser().resolve()
    else:
        config_path = Path(config_path).resolve()
        try:
            config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError) as error:
            raise ValueError(f'Cannot read dataset configuration {config_path}: {error}') from error
        value = config.get('dataset_root') if isinstance(config, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{config_path} must contain a nonempty dataset_root string')
        root = Path(value).expanduser()
        if not root.is_absolute():
            raise ValueError(f'dataset_root in {config_path} must be an absolute path')
        root = root.resolve()
    if not (root / 'dataset.json').is_file():
        raise ValueError(f'No dataset.json in configured dataset root: {root}')
    return root
