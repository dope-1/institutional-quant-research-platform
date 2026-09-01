from __future__ import annotations

from typing import Annotated

from fastapi import Request
from fastapi.params import Depends

from quant_platform.api.artifacts import ArtifactRepository


def get_artifact_repository(request: Request) -> ArtifactRepository:
    return request.app.state.artifact_repository


ArtifactRepositoryDependency = Annotated[ArtifactRepository, Depends(get_artifact_repository)]
