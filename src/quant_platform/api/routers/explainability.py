from fastapi import APIRouter

from quant_platform.api.dependencies import ArtifactRepositoryDependency
from quant_platform.api.schemas import ExplainabilityResponse

router = APIRouter(tags=["research"])


@router.get("/explainability", response_model=ExplainabilityResponse)
def get_explainability(
    repository: ArtifactRepositoryDependency,
) -> ExplainabilityResponse:
    return repository.explainability()
