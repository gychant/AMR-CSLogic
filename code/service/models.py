""""Models defined for the services"""


class PredicateCalculus(object):
    def __init__(self, predicate, arguments):
        self.predicate = predicate
        self.arguments = arguments

    def __repr__(self):
        return self.predicate + "(" + ", ".join(self.arguments) + ")"

    def to_json(self):
        return {
            "predicate": self.predicate,
            "arguments": self.arguments
        }


class DataValidationError(Exception):
    """Used for an data validation errors when deserializing"""
    pass

