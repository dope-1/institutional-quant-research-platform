from fastapi import APIRouter

from quant_platform.api.dependencies import ArtifactRepositoryDependency
from quant_platform.api.schemas import RegimesResponse

router = APIRouter(tags=["research"])


@router.get("/regimes", response_model=RegimesResponse)
def get_regimes(
    repository: ArtifactRepositoryDependency,
) -> RegimesResponse:
    return repository.regimes()
