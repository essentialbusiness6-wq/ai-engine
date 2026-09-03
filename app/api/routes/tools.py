"""
GET /v1/tools

Lets a consuming application introspect which tools this AI Engine
instance currently exposes -- useful for building admin UIs or validating
that a deployment has the expected capabilities wired up.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import get_container, get_principal
from app.domain.interfaces import Principal

router = APIRouter(prefix="/v1", tags=["tools"])


class ToolInfo(BaseModel):
    name: str
    description: str
    required_permission: str | None
    parameters: list[dict]


@router.get("/tools", response_model=list[ToolInfo])
def list_tools(
    principal: Principal = Depends(get_principal),
    container=Depends(get_container),
) -> list[ToolInfo]:
    definitions = container.tool_registry.list_definitions()
    return [
        ToolInfo(
            name=d.name,
            description=d.description,
            required_permission=d.required_permission,
            parameters=[
                {"name": p.name, "type": p.type, "required": p.required} for p in d.parameters
            ],
        )
        for d in definitions
    ]
