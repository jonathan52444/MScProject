"""
nested.py
"""

from datetime import datetime

def get_nested(obj, dotted):
    cur = obj
    for tok in dotted.split('.'):
        if cur is None:
            return None
        if '[' in tok:                           # list index
            key, idx = tok[:-1].split('[')
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
            if not isinstance(cur, list):
                return None
            try:
                cur = cur[int(idx)]
            except (ValueError, IndexError):
                return None
        else:                                    # plain dict key
            if not isinstance(cur, dict):
                return None
            cur = cur.get(tok)
    return cur

def value_from_paths(study, path_string):
    """
    One path  -> scalar via get_nested
    Two paths -> if both ISO-like dates, return (date2 - date1).days
    Else      -> None
    """
    parts = [p.strip() for p in path_string.split('|')]
    if len(parts) == 1:
        return get_nested(study, parts[0])
    if len(parts) == 2:
        d1 = get_nested(study, parts[0])
        d2 = get_nested(study, parts[1])
        if d1 and d2:
            try:
                d1 = datetime.fromisoformat(d1.split('T')[0])
                d2 = datetime.fromisoformat(d2.split('T')[0])
                return (d2 - d1).days
            except ValueError:
                pass
    return None
