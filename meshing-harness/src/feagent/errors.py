"""Exception types for feagent, and frontend-aware wording for their hints."""
import os

FRONTEND_ENV = "FEAGENT_FRONTEND"  # "mcp" when the CLI runs under the MCP server


def for_frontend(cli: str, mcp: str) -> str:
    """``mcp`` when running under the MCP server ($FEAGENT_FRONTEND=mcp), else ``cli``: hints
    name the tool calls the agent can actually make."""
    return mcp if os.environ.get(FRONTEND_ENV) == "mcp" else cli


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
