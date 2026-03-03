import pytest

import intermine314.service.tor as tor
from intermine314.service.errors import TorConfigurationError


def test_tor_service_defaults_to_strict_dns_safe_proxy_scheme():
    with pytest.raises(TorConfigurationError, match="socks5h"):
        tor.tor_service("https://example.org/service", scheme="socks5")
