from fastapi import APIRouter

from quant_platform.api.dependencies import ArtifactRepositoryDependency
from quant_platform.api.schemas import ModelsResponse

router = APIRouter(tags=["research"])


@router.get("/models", response_model=ModelsResponse)
def get_models(
    repository: ArtifactRepositoryDependency,
) -> ModelsResponse:
    return repository.models()
