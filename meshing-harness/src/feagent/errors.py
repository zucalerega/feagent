"""Exception types for feagent."""


class FeagentError(Exception):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message} (hint: {self.hint})" if self.hint else self.message


class CadLoadError(FeagentError):
    pass


class QueryError(FeagentError):
    pass


class SpecError(FeagentError):
    """Message starts with the dotted field path, e.g. "groups.fixed.expect: ..."."""
