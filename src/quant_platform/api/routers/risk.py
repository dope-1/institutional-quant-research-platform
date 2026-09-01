from fastapi import APIRouter

from quant_platform.api.dependencies import ArtifactRepositoryDependency
from quant_platform.api.schemas import RiskResponse

router = APIRouter(tags=["research"])


@router.get("/risk", response_model=RiskResponse)
def get_risk(
    repository: ArtifactRepositoryDependency,
) -> RiskResponse:
    return repository.risk()
