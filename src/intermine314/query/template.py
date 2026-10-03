"""Template construction and constraint state.

Wrapper XML, named-template endpoints and adjusted execution are implemented
in the next compatibility stage; inherited Query execution remains ordinary.
"""

from intermine314.query.builder import Query
from intermine314.query.constraints import TemplateConstraintFactory


class Template(Query):
    """A query using editable, optionally switchable typed constraints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_factory = TemplateConstraintFactory(compatibility=self.compatibility)

    @property
    def editable_constraints(self):
        return [constraint for constraint in self.constraints if constraint.editable]
