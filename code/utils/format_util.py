"""
Utility functions for formatting
"""
import json
import inspect


def to_json(obj):
    if isinstance(obj, dict):
        results = {}
        for key in obj:
            results[key] = to_json(obj[key])
        return results
    elif isinstance(obj, list):
        results = []
        for elem in obj:
            results.append(to_json(elem))
        return results
    elif hasattr(obj, "to_json") and inspect.ismethod(getattr(obj, "to_json")):
        return obj.to_json()
    else:
        return obj

