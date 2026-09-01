from fastapi import APIRouter

from quant_platform.api.dependencies import ArtifactRepositoryDependency
from quant_platform.api.schemas import OptimizationResponse

router = APIRouter(tags=["research"])


@router.get("/optimization", response_model=OptimizationResponse)
def get_optimization(
    repository: ArtifactRepositoryDependency,
) -> OptimizationResponse:
    return repository.optimization()
