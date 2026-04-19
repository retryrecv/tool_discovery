from __future__ import annotations

import pytest

from planner import CapabilitySpec, FilePlannerStore, PlannerRuntime, RegistryActionGrounder, RuleBasedPlanningPolicy


@pytest.fixture
def runtime(tmp_path):
    grounder = RegistryActionGrounder(
        capabilities=(
            CapabilitySpec("web_search", "Search the web for information", ("search", "research", "find")),
            CapabilitySpec("text_generation", "Write drafts and summaries", ("write", "draft", "summarize", "recommend")),
            CapabilitySpec("workspace_action", "Perform workspace actions", ("build", "create", "run", "update")),
            CapabilitySpec("validator", "Validate or compare intermediate results", ("validate", "check", "compare")),
        )
    )
    return PlannerRuntime(
        FilePlannerStore(tmp_path / "planner-store"),
        policy=RuleBasedPlanningPolicy(),
        action_grounder=grounder,
    )
