from intermine314.webservice import Registry, Service

"""
Functions for making use of registry data
================================================

"""


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
        registry = Registry()
        info = registry.info(mine)
        return {
            "API Version:": info.get("api_version"),
            "Release Version:": info.get("release_version"),
            "InterMine Version:": info.get("intermine_version"),
        }
    except KeyError:
        return "No such mine available"


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
        registry = Registry()
        info = registry.info(mine)
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
    except KeyError:
        return "No such mine available"


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
        registry = Registry()
        service_root = registry.service_root(mine)
        service = Service(service_root)
        query = service.new_query("DataSet")
        query.add_view("name", "url")
        list = []

        for row in query.rows():
            try:
                list.append(row["name"])

            except KeyError:
                print("No info available")
        list.sort()
        for i in range(len(list)):
            print("Name: " + list[i])
        return None
    except KeyError:
        return "No such mine available"


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
    count = 0
    mines = Service.get_all_mines(organism=organism)
    for i in range(len(mines)):
        if organism is None:
            print(mines[i]["name"])
            count = count + 1
        else:
            print(mines[i]["name"])
            count = count + 1
    if count == 0:
        return "No such mine available"
    else:
        return None
