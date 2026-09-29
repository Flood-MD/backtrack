"""User configuration: LLM provider profiles, contact email, workspace location.

Config lives in ``$BACKTRACK_HOME`` (default ``~/.backtrack``) as ``config.json``
with mode 0600. API keys can be stored there or referenced by environment
variable name (``api_key_env``); environment variables always win when set.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# kind "openai" = any OpenAI-compatible /chat/completions API; "anthropic" = Messages API.
PRESETS: dict[str, dict] = {
    "openai": {
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o",
        "token_param": "max_completion_tokens",
    },
    "openrouter": {
        "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "openai/gpt-4o",
        "token_param": "max_tokens",
    },
    "huggingface": {
        "kind": "openai",
        "base_url": "https://router.huggingface.co/v1",
        "api_key_env": "HF_TOKEN",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "token_param": "max_tokens",
    },
    "anthropic": {
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-5",
        "token_param": "max_tokens",
    },
    # Any other OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, ...).
    "custom": {
        "kind": "openai",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "",
        "model": "",
        "token_param": "max_tokens",
    },
}


@dataclass
class ProviderConfig:
    name: str
    kind: str = "openai"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    api_key_env: str = ""
    token_param: str = "max_tokens"
    extra_headers: dict[str, str] = field(default_factory=dict)

    def resolve_key(self) -> str:
        if self.api_key_env and os.environ.get(self.api_key_env):
            return os.environ[self.api_key_env]
        return self.api_key


@dataclass
class Config:
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    default_provider: str = ""
    # Polite-pool / Unpaywall contact address. Unpaywall requires a real one.
    contact_email: str = ""
    semantic_scholar_key: str = ""
    workspace: str = ""
    citation_depth: int = 1
    max_papers: int = 40
    max_refs_per_paper: int = 25

    def provider(self, name: str | None = None) -> ProviderConfig:
        name = name or self.default_provider
        if not name or name not in self.providers:
            raise KeyError(
                f"No LLM provider {'%r ' % name if name else ''}configured. "
                "Run `backtrack provider add <openai|openrouter|huggingface|anthropic|custom>`."
            )
        return self.providers[name]


def home_dir() -> Path:
    return Path(os.environ.get("BACKTRACK_HOME") or Path.home() / ".backtrack")


def config_path() -> Path:
    return home_dir() / "config.json"


def workspace_dir(cfg: Config | None = None) -> Path:
    if os.environ.get("BACKTRACK_WORKSPACE"):
        return Path(os.environ["BACKTRACK_WORKSPACE"])
    if cfg and cfg.workspace:
        return Path(cfg.workspace)
    return home_dir() / "workspace"


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return Config()
    raw = json.loads(path.read_text())
    providers = {n: ProviderConfig(name=n, **p) for n, p in raw.pop("providers", {}).items()}
    known = {k: v for k, v in raw.items() if k in Config.__dataclass_fields__}
    return Config(providers=providers, **known)


def save_config(cfg: Config, path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg)
    for n, p in data["providers"].items():
        p.pop("name", None)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return path


def new_provider(preset: str, name: str | None = None, **overrides) -> ProviderConfig:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
    data = {**PRESETS[preset], **{k: v for k, v in overrides.items() if v}}
    return ProviderConfig(name=name or preset, **data)
