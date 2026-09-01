from fastapi import APIRouter

from quant_platform.api.dependencies import ArtifactRepositoryDependency
from quant_platform.api.schemas import MethodologyResponse

router = APIRouter(tags=["research"])


@router.get("/methodology", response_model=MethodologyResponse)
def get_methodology(
    repository: ArtifactRepositoryDependency,
) -> MethodologyResponse:
    return repository.methodology()
