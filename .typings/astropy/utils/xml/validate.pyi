def validate_schema(filename, schema_file):
    """
    Validates an XML file against a schema or DTD.

    Parameters
    ----------
    filename : str
        The path to the XML file to validate

    schema_file : str
        The path to the XML schema or DTD

    Returns
    -------
    returncode, stdout, stderr : int, str, str
        Returns the returncode from xmllint and the stdout and stderr
        as strings
    """
