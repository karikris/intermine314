import base64


def build_basic_auth_header(username: str, password: str) -> str:
    payload = f"{username}:{password}".encode("utf-8")
    return "Basic " + base64.b64encode(payload).decode("ascii")


def build_token_auth_header(token: str) -> str:
    return f"Token {token}"
