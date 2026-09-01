from fastapi import APIRouter

from quant_platform.api.dependencies import ArtifactRepositoryDependency
from quant_platform.api.schemas import OverviewResponse

router = APIRouter(tags=["research"])


@router.get("/overview", response_model=OverviewResponse)
def get_overview(
    repository: ArtifactRepositoryDependency,
) -> OverviewResponse:
    return repository.overview()
