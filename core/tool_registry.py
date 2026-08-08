import inspect
from typing import Callable, Any, Dict, List
from pydantic import create_model
from core.logging import get_logger


logger = get_logger("tool_registry")


class ToolRegistry:
    """Registry for managing and executing agent tools and generating JSON schemas."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(self, func: Callable) -> Callable:
        """Decorator/method to register a Python function as an agent tool."""
        name = func.__name__
        description = (func.__doc__ or "No description provided.").strip()

        # Dynamically build a Pydantic model from the function's type hints
        sig = inspect.signature(func)
        fields = {}
        for param_name, param in sig.parameters.items():
            annotation = (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else Any
            )
            default = (
                param.default
                if param.default != inspect.Parameter.empty
                else ...
            )
            fields[param_name] = (annotation, default)

        # Generate standard OpenAPI/JSON Schema via Pydantic
        pydantic_model = create_model(f"{name}_input", **fields)
        json_schema = pydantic_model.model_json_schema()

        # Clean schema for Groq / OpenAI function specs
        parameters = {
            "type": "object",
            "properties": json_schema.get("properties", {}),
            "required": json_schema.get("required", []),
        }

        tool_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

        self._tools[name] = func
        self._schemas.append(tool_schema)
        logger.debug("Registered tool %s", name)
        return func

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Returns all registered tool schemas formatted for the LLM payload."""
        return self._schemas

    def execute(self, name: str, kwargs: Dict[str, Any]) -> str:
        """Executes a registered tool by name with keyword arguments."""
        if name not in self._tools:
            logger.warning("Attempted to execute unregistered tool %s", name)
            return f"Error: Tool '{name}' is not registered."

        logger.info("Executing registered tool %s with kwargs %s", name, kwargs)
        try:
            result = self._tools[name](**kwargs)
            logger.debug("Tool %s completed successfully", name)
            return str(result)
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return f"Error executing tool '{name}': {str(e)}"