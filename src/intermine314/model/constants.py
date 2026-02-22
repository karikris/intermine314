_MODEL_LOG_PREVIEW_CHARS = 160
_MODEL_HASH_PREFIX_CHARS = 12
_MODEL_HASH_TEXT_CHUNK_CHARS = 4096
_PATH_SEGMENT_CACHE_MAXSIZE = 4096

# Frequently repeated model/XML literals.
_XML_TAG_MODEL = "model"
_XML_TAG_CLASS = "class"
_XML_TAG_ATTRIBUTE = "attribute"
_XML_TAG_REFERENCE = "reference"
_XML_TAG_COLLECTION = "collection"
_XML_ATTR_NAME = "name"
_XML_ATTR_PACKAGE = "package"
_XML_ATTR_TYPE = "type"
_XML_ATTR_REFERENCED_TYPE = "referenced-type"
_XML_ATTR_REVERSE_REFERENCE = "reverse-reference"
_XML_ATTR_EXTENDS = "extends"
_XML_ATTR_IS_INTERFACE = "is-interface"
_XML_VALUE_TRUE = "true"
_MODEL_PARSE_ERROR_MESSAGE = "Error parsing model"

# Core model literals used in multiple places.
_ROOT_OBJECT_CLASS = "Object"
_ID_FIELD_NAME = "id"
_ID_FIELD_TYPE = "Integer"

# Column operator literals reused across helpers.
_OP_LOOKUP = "LOOKUP"
_OP_IN = "IN"
_OP_NOT_IN = "NOT IN"
_OP_ONE_OF = "ONE OF"
_OP_NONE_OF = "NONE OF"
_OP_IS_NULL = "IS NULL"
_OP_IS_NOT_NULL = "IS NOT NULL"
_OP_IS = "IS"
_OP_IS_NOT = "IS NOT"

NUMERIC_TYPES = frozenset(
    ("int", "Integer", "float", "Float", "double", "Double", "long", "Long", "short", "Short")
)
NUMERIC_TYPES_NORMALIZED = frozenset(("int", "integer", "float", "double", "long", "short"))
