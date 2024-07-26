from pathlib import Path

from .config_workspace import WorkspaceConfig


def test_config_workspace_model_dir() -> None:
    config = WorkspaceConfig(model_dir=Path("model_dir"))
    assert config.model_dir == Path("model_dir")


def test_config_workspace_checkpoint_dir() -> None:
    config = WorkspaceConfig(
        model_dir=Path("model_dir"), checkpoint_dir=Path("checkpoint_dir")
    )
    assert config.checkpoint_dir == Path("checkpoint_dir")


def test_config_workspace_logging_dir() -> None:
    config = WorkspaceConfig(
        model_dir=Path("model_dir"), logging_dir=Path("logging_dir")
    )
    assert config.logging_dir == Path("logging_dir")
