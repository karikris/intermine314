from typing import Dict, Mapping, Optional


def _copy_subclasses(subclasses: Optional[Mapping[str, str]]) -> Dict[str, str]:
    if subclasses is None:
        return {}
    return dict(subclasses)
