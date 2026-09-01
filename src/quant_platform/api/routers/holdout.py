from fastapi import APIRouter

from quant_platform.api.dependencies import ArtifactRepositoryDependency
from quant_platform.api.schemas import HoldoutResponse

router = APIRouter(tags=["research"])


@router.get("/holdout", response_model=HoldoutResponse)
def get_holdout(
    repository: ArtifactRepositoryDependency,
) -> HoldoutResponse:
    return repository.holdout()
