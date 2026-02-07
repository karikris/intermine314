from intermine314.webservice import Registry, Service

"""
Functions for making use of registry data
================================================

"""

NO_SUCH_MINE = "No such mine available"


def _safe_registry_info(mine):
    registry = Registry()
    info = registry.info(mine)
    return registry, info


def getVersion(mine):
    """
    A function to return the API version, release version and
    InterMine version numbers
    ================================================
    example:

        >>> from intermine314 import registry
        >>> registry.getVersion('flymine')
        >>> {'API Version:': '30', 'Release Version:': '48 2019 October',
        'InterMine Version:': '4.1.0'}

    """
    try:
        _, info = _safe_registry_info(mine)
        return {
            "API Version:": info.get("api_version"),
            "Release Version:": info.get("release_version"),
            "InterMine Version:": info.get("intermine_version"),
        }
    except Exception:
        return NO_SUCH_MINE


def getInfo(mine):
    """
    A function to get information about a mine
    ================================================
    example:

        >>> from intermine314 import registry
        >>> registry.getInfo('flymine')
        Description:  An integrated database for Drosophila genomics
        URL: https://www.flymine.org/flymine
        API Version: 25
        Release Version: 45.1 2017 August
        InterMine Version: 1.8.5
        Organisms:
        D. melanogaster
        Neighbours:
        MODs

    """
    try:
        _, info = _safe_registry_info(mine)
        print("Description: " + (info.get("description") or ""))
        print("URL: " + (info.get("url") or ""))
        print("API Version: " + (info.get("api_version") or ""))
        print("Release Version: " + (info.get("release_version") or ""))
        print("InterMine Version: " + (info.get("intermine_version") or ""))
        print("Organisms: ")
        for organism in info.get("organisms", []):
            print(organism)
        print("Neighbours: ")
        for neighbour in info.get("neighbours", []):
            print(neighbour)
        return None
    except Exception:
        return NO_SUCH_MINE


def getData(mine):
    """
    A function to get datasets corresponding to a mine
    ================================================
    example:

        >>> from intermine314 import registry
        >>> registry.getData('flymine')
        Name: Affymetrix array: Drosophila1
        Name: Affymetrix array: Drosophila2
        Name: Affymetrix array: GeneChip Drosophila Genome 2.0 Array
        Name: Affymetrix array: GeneChip Drosophila Genome Array
        Name: Anoph-Expr data set
        Name: BDGP cDNA clone data set.....


    """
    try:
        registry, info = _safe_registry_info(mine)
        service_root = info.get("url") or registry.service_root(mine)
        if not service_root:
            return NO_SUCH_MINE

        service = Service(service_root)
        dataset_names = []
        query_shapes = (
            ("DataSet", ("DataSet.name", "DataSet.url"), ("DataSet.name", "name")),
            ("Dataset", ("Dataset.name", "Dataset.url"), ("Dataset.name", "name")),
        )
        for class_name, views, keys in query_shapes:
            try:
                query = service.new_query(class_name)
                query.add_view(*views)
                for row in query.rows(row="dict", start=0, size=500):
                    value = None
                    for key in keys:
                        if key in row and row[key]:
                            value = row[key]
                            break
                    if value:
                        dataset_names.append(str(value))
                break
            except Exception:
                continue

        for name in sorted(set(dataset_names)):
            print("Name: " + name)
        return None
    except Exception:
        return NO_SUCH_MINE


def getMines(organism=None):
    """
    A function to get mines containing the organism
    ================================================
    example:

        >>> from intermine314 import registry
        >>> registry.getMines('D. melanogaster')
        FlyMine
        FlyMine Beta
        XenMine

    """
    try:
        mines = Service.get_all_mines(organism=organism)
    except Exception:
        return NO_SUCH_MINE

    names = sorted(set(m.get("name") for m in mines if isinstance(m, dict) and m.get("name")))
    if not names:
        return NO_SUCH_MINE
    for name in names:
        print(name)
    return None
