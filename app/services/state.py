"""Atomic project state transitions (v4 §9.4).

The check-then-set pattern (read state, decide, write new state) has a race
window: two concurrent callers can both pass the guard and then both run the
destructive ``reset_dirs``. ``transition_if`` instead performs a single
conditional UPDATE — ``SET state=:new WHERE id=:id AND state IN (:allowed)`` —
which the database applies atomically, so exactly one caller wins. The loser
sees ``False`` and the caller raises 409 CONFLICT_BUSY.
"""
from app.db import models


def transition_if(db, project, allowed: set[str], new_state: str) -> bool:
    """Move ``project.state`` to ``new_state`` iff it is currently in ``allowed``.

    Returns True if this caller performed the transition, False otherwise. On
    success the in-session ``project`` is refreshed to reflect the new state.
    """
    affected = (
        db.query(models.Project)
        .filter(models.Project.id == project.id, models.Project.state.in_(allowed))
        .update(
            {models.Project.state: new_state, models.Project.error: None},
            synchronize_session=False,
        )
    )
    db.commit()
    if affected:
        db.refresh(project)
    return bool(affected)
