"""Historical registry helpers over the configured, managed Registry transport.

Adapted from the InterMine Python client 1.13.0 under LICENSE-BSD and NOTICE.
"""

from __future__ import annotations

import json
from contextlib import closing, contextmanager

__all__ = ['getVersion', 'getInfo', 'getData', 'getMines']


@contextmanager
def _registry_client(registry, registry_options):
    if registry is not None:
        if registry_options:
            raise TypeError('registry_options cannot be combined with an injected registry')
        yield registry
    else:
        from intermine314.webservice import Registry

        with Registry(**registry_options) as client:
            yield client


def _instance(client, mine):
    # Preserve source case and direct path concatenation for detail requests.
    with closing(client._opener.open(client._list_url() + '/' + mine)) as response:
        payload = response.read()
    return json.loads(payload)['instance']


def getVersion(mine, *, registry=None, **registry_options):
    """Return historical version-label keys, or ``'No such mine available'``.

    Keyword options configure an owned legacy Registry. An injected Registry is
    borrowed, retaining its configuration, service cache and lifetime.
    """
    with _registry_client(registry, registry_options) as client:
        try:
            info = _instance(client, mine)
            return {
                'API Version:': info['api_version'],
                'Release Version:': info['release_version'],
                'InterMine Version:': info['intermine_version'],
            }
        except KeyError:
            return 'No such mine available'


def getInfo(mine, *, registry=None, **registry_options):
    """Print source labels and field order; return None or the missing message."""
    with _registry_client(registry, registry_options) as client:
        try:
            info = _instance(client, mine)
            for label, key in [('Description', 'description'), ('URL', 'url'),
                               ('API Version', 'api_version'), ('Release Version', 'release_version'),
                               ('InterMine Version', 'intermine_version')]:
                print(label + ': ' + info[key])
            print('Organisms: ')
            for organism in info['organisms']:
                print(organism)
            print('Neighbours: ')
            for neighbour in info['neighbours']:
                print(neighbour)
            return None
        except KeyError:
            return 'No such mine available'


def getData(mine, *, registry=None, **registry_options):
    """Print sorted dataset names using the detail URL and Registry transport.

    The locally created Service and result stream are owned; its session is
    borrowed from the Registry. Existing cached Service clients are untouched.
    """
    with _registry_client(registry, registry_options) as client:
        try:
            info = _instance(client, mine)
            with client._new_service(
                info['url'], request_timeout=client.request_timeout,
                proxy_url=client.proxy_url, session=client._session,
                verify_tls=client.verify_tls, tor=client.tor,
                strict_tor_proxy_scheme=client.strict_tor_proxy_scheme,
                allow_insecure_tor_proxy_scheme=client.allow_insecure_tor_proxy_scheme,
                allow_http_over_tor=client.allow_http_over_tor,
                user_agent=client.user_agent, compatibility=client.compatibility,
            ) as service:
                query = service.new_query('DataSet')
                query.select('name', 'url')
                names = []
                with closing(query.rows(row='rr')) as rows:
                    for row in rows:
                        try:
                            names.append(row['name'])
                        except KeyError:
                            print('No info available')
                names.sort()
                for name in names:
                    print('Name: ' + name)
                return None
        except KeyError:
            return 'No such mine available'


def getMines(organism=None, *, registry=None, **registry_options):
    """Print mine names with source exact/one-leading-space organism matching."""
    with _registry_client(registry, registry_options) as client:
        count = 0
        for mine in client.all_mines():
            if organism is None:
                print(mine['name'])
                count += 1
            else:
                for entry in mine['organisms']:
                    if entry == organism or entry == ' ' + organism:
                        print(mine['name'])
                        count += 1
        return None if count else 'No such mine available'
