from app.schemas.project import OrthoOut, ProjectOut


def serialize_project(project) -> ProjectOut:
    """ORM Project -> API response model."""
    return ProjectOut(
        project_id=project.id,
        name=project.name,
        model_key=project.model_key,
        state=project.state,
        source_epsg=project.source_epsg,
        params=project.params or {},
        recommended_k=project.recommended_k,
        available_k=project.available_k,
        current_run=project.current_run or 1,
        runs=project.runs or [],
        orthos=[OrthoOut.model_validate(o) for o in project.orthos],
        error=project.error,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )
