"""Serialize identifiers for the server's linewise, double-quoted tokenizer."""


def quote_identifier(value):
    value = str(value)
    if "\r" in value or "\n" in value:
        raise ValueError("Identifiers cannot contain line breaks")
    return '"' + value.replace('"', '""') + '"'
