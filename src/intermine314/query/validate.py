def validate_view_path(model, path: str) -> bool:
    model.make_path(path)
    return True


def validate_view_paths(model, paths) -> bool:
    for path in paths:
        validate_view_path(model, path)
    return True
