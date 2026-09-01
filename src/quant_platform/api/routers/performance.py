from fastapi import APIRouter

from quant_platform.api.dependencies import ArtifactRepositoryDependency
from quant_platform.api.schemas import PerformanceResponse, RankICResponse

router = APIRouter(tags=["research"])


@router.get("/performance", response_model=PerformanceResponse)
def get_performance(
    repository: ArtifactRepositoryDependency,
) -> PerformanceResponse:
    return repository.performance()


@router.get("/rank-ic", response_model=RankICResponse)
def get_rank_ic(
    repository: ArtifactRepositoryDependency,
) -> RankICResponse:
    return repository.rank_ic()
