intermine314 package
====================

intermine314.service
--------------------

Package-managed sessions retry explicitly read-only operations, including query
POSTs and metadata reads. Mutations and unknown operations do not automatically
retry, even when their HTTP method is GET. A failed write can already have taken
effect on the server; inspect server state before retrying it yourself.

When you supply a Requests session, its adapters and retry configuration remain
under your control. The package does not override that policy or close the
borrowed session. Avoid configuring automatic retries of non-idempotent writes.
Cloned package openers retain the managed retry policy while borrowing their
parent's session. Concurrent requests select fixed policies without changing
shared adapter settings.

Service metadata and template XML use the configured HTTP opener. Whole-response
and sized reads both decode supported HTTP content encodings, including gzip,
deflate, and Zstandard on Python 3.14. End-of-stream and read/decompression errors
close the response; a borrowed session remains open. Metadata cache behavior,
authentication, TLS verification, proxy configuration, and timeouts are shared
by the native and legacy facades.

.. automodule:: intermine314.service
   :members:
   :undoc-members:
   :show-inheritance:

intermine314.webservice
-----------------------

Historical re-exports remain available, including ``Query``, ``Template``,
``Model``, ``ListManager``, ``ServiceError``, ``WebserviceError``,
``InterMineURLOpener``, ``ResultIterator``, ``idresolution`` and
``requires_version``. These aliases share their canonical implementations;
API details are indexed once under those implementations. The query, results,
constraints, pathfeatures and list modules also retain their historical owned
bindings and public path/version constants.

.. automodule:: intermine314.webservice
   :members:
   :exclude-members: Attribute, Collection, Column, InterMineURLOpener, ListManager, Model, Query, Reference, ResultIterator, ServiceError, Template, WebserviceError, idresolution, requires_version
   :undoc-members:
   :show-inheritance:

intermine314.query
------------------

.. automodule:: intermine314.query
   :members:
   :exclude-members: Class, Column, ConstraintNode, Join, Model, PathDescription, ReadableException, Reference, SortOrder, SortOrderList, constraints, openAnything
   :undoc-members:
   :show-inheritance:

intermine314.export
-------------------

.. automodule:: intermine314.export
   :members:
   :undoc-members:
   :show-inheritance:

intermine314.model
------------------

.. automodule:: intermine314.model
   :members:
   :undoc-members:
   :show-inheritance:

intermine314.results
--------------------

.. automodule:: intermine314.results
   :members:
   :exclude-members: Attribute, Collection, Reference, VERSION, WebserviceError
   :undoc-members:
   :show-inheritance:

intermine314.pathfeatures
-------------------------

.. automodule:: intermine314.pathfeatures
   :members:
   :exclude-members: PATH_PATTERN, PATTERN_STR
   :undoc-members:
   :show-inheritance:

intermine314.constraints
------------------------

.. automodule:: intermine314.constraints
   :members:
   :exclude-members: PATH_PATTERN, PathFeature, ReadableException
   :undoc-members:
   :show-inheritance:

intermine314.errors
-------------------

.. automodule:: intermine314.errors
   :members:
   :undoc-members:
   :show-inheritance:

intermine314.lists
------------------

.. automodule:: intermine314.lists
   :members:
   :undoc-members:
   :show-inheritance:

intermine314.idresolution
-------------------------

.. automodule:: intermine314.idresolution
   :members:
   :undoc-members:
   :show-inheritance:

intermine314.registry
---------------------

.. automodule:: intermine314.registry
   :members:
   :undoc-members:
   :show-inheritance:

intermine314.query_manager
--------------------------

.. automodule:: intermine314.query_manager
   :members:
   :undoc-members:
   :show-inheritance:

intermine314.bar_chart
----------------------

.. automodule:: intermine314.bar_chart
   :members:
   :undoc-members:
   :show-inheritance:

intermine314.decorators
-----------------------

.. automodule:: intermine314.decorators
   :members:
   :undoc-members:
   :show-inheritance:

intermine314.util
-----------------

.. automodule:: intermine314.util
   :members:
   :undoc-members:
   :show-inheritance:

Package contents
----------------

.. automodule:: intermine314
   :members:
