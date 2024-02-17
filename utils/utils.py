from typing import Tuple

def str2tuple(string: str) -> Tuple:
    """ Convert string to tuple

    Args:
        string: The string to be converted

    Returns:
        The tuple converted from the string
    """

    string=string.replace('(','').replace(')','')
    return tuple(map(int, string.split(',')))
