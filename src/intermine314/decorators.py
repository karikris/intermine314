"""Version guards for service methods."""

from functools import wraps

__all__ = ["requires_version"]


def requires_version(required):
    """Require a service version, preserving the decorated method's metadata."""

    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            if self.version < required:
                from intermine314.errors import ServiceError

                raise ServiceError(
                    f"Service must be at version {required}, but is at {self.version}"
                )
            return function(self, *args, **kwargs)

        return wrapper

    return decorator
