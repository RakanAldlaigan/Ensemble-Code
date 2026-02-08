from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import aiohttp
from prompt_toolkit.shortcuts.choice_input import ChoiceInput
from pydantic import SecretStr

from kimi_cli.auth.platforms import get_platform_name_for_provider, refresh_managed_models
from kimi_cli.cli import Reload, SwitchToWeb
from kimi_cli.config import LLMModel, LLMProvider, load_config, save_config
from kimi_cli.exception import ConfigError
from kimi_cli.session import Session
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.ui.shell.console import console
from kimi_cli.utils.aiohttp import new_client_session
from kimi_cli.utils.changelog import CHANGELOG
from kimi_cli.utils.datetime import format_relative_time
from kimi_cli.utils.slashcmd import SlashCommand, SlashCommandRegistry

if TYPE_CHECKING:
    from kimi_cli.config import Config
    from kimi_cli.ui.shell import Shell

type ShellSlashCmdFunc = Callable[[Shell, str], None | Awaitable[None]]
"""
A function that runs as a Shell-level slash command.

Raises:
    Reload: When the configuration should be reloaded.
"""


registry = SlashCommandRegistry[ShellSlashCmdFunc]()
shell_mode_registry = SlashCommandRegistry[ShellSlashCmdFunc]()
OPENAI_PROVIDER_KEY = "openai"
OPENAI_MODEL_PREFIX = "openai/"


def _ensure_kimi_soul(app: Shell) -> KimiSoul | None:
    if not isinstance(app.soul, KimiSoul):
        console.print("[red]KimiSoul required[/red]")
        return None
    return app.soul


@registry.command(aliases=["quit"])
@shell_mode_registry.command(aliases=["quit"])
def exit(app: Shell, args: str):
    """Exit the application"""
    # should be handled by `Shell`
    raise NotImplementedError


SKILL_COMMAND_PREFIX = "skill:"

_KEYBOARD_SHORTCUTS = [
    ("Ctrl-X", "Toggle agent/shell mode"),
    ("Ctrl-J / Alt-Enter", "Insert newline"),
    ("Ctrl-V", "Paste (supports images)"),
    ("Ctrl-D", "Exit"),
    ("Ctrl-C", "Interrupt"),
]


@registry.command(aliases=["h", "?"])
@shell_mode_registry.command(aliases=["h", "?"])
def help(app: Shell, args: str):
    """Show help information"""
    from rich.console import Group, RenderableType
    from rich.text import Text

    from kimi_cli.utils.rich.columns import BulletColumns

    def section(title: str, items: list[tuple[str, str]], color: str) -> BulletColumns:
        lines: list[RenderableType] = [Text.from_markup(f"[bold]{title}:[/bold]")]
        for name, desc in items:
            lines.append(
                BulletColumns(
                    Text.from_markup(f"[{color}]{name}[/{color}]: [grey50]{desc}[/grey50]"),
                    bullet_style=color,
                )
            )
        return BulletColumns(Group(*lines))

    renderables: list[RenderableType] = []
    renderables.append(
        BulletColumns(
            Group(
                Text.from_markup("[grey50]Help! I need somebody. Help! Not just anybody.[/grey50]"),
                Text.from_markup("[grey50]Help! You know I need someone. Help![/grey50]"),
                Text.from_markup("[grey50]\u2015 The Beatles, [italic]Help![/italic][/grey50]"),
            ),
            bullet_style="grey50",
        )
    )
    renderables.append(
        BulletColumns(
            Text(
                "Sure, Kimi is ready to help! "
                "Just send me messages and I will help you get things done!"
            ),
        )
    )

    commands: list[SlashCommand[Any]] = []
    skills: list[SlashCommand[Any]] = []
    for cmd in app.available_slash_commands.values():
        if cmd.name.startswith(SKILL_COMMAND_PREFIX):
            skills.append(cmd)
        else:
            commands.append(cmd)

    renderables.append(section("Keyboard shortcuts", _KEYBOARD_SHORTCUTS, "yellow"))
    renderables.append(
        section(
            "Slash commands",
            [(c.slash_name(), c.description) for c in sorted(commands, key=lambda c: c.name)],
            "blue",
        )
    )
    if skills:
        renderables.append(
            section(
                "Skills",
                [(c.slash_name(), c.description) for c in sorted(skills, key=lambda c: c.name)],
                "cyan",
            )
        )

    with console.pager(styles=True):
        console.print(Group(*renderables))


@registry.command
@shell_mode_registry.command
def version(app: Shell, args: str):
    """Show version information"""
    from kimi_cli.constant import VERSION

    console.print(f"kimi, version {VERSION}")


def _is_openai_chat_model(model_id: str) -> bool:
    model_id = model_id.lower()
    return (
        model_id.startswith("gpt-")
        or model_id.startswith("o1")
        or model_id.startswith("o3")
        or model_id.startswith("o4")
        or model_id.startswith("chatgpt")
    )


async def _fetch_openai_chat_models(*, base_url: str, api_key: str) -> list[str]:
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        async with (
            new_client_session() as session,
            session.get(
                models_url,
                headers={"Authorization": f"Bearer {api_key}"},
                raise_for_status=True,
            ) as resp,
        ):
            payload: object = await resp.json()
    except aiohttp.ClientError:
        raise

    if not isinstance(payload, dict):
        raise ValueError("Unexpected response from OpenAI models endpoint.")
    raw_data = payload.get("data")
    if not isinstance(raw_data, list):
        raise ValueError("Unexpected response from OpenAI models endpoint.")

    model_ids = sorted(
        {
            str(item.get("id"))
            for item in raw_data
            if isinstance(item, dict) and item.get("id") and _is_openai_chat_model(str(item["id"]))
        }
    )
    return model_ids


def _upsert_openai_models(
    config: Config, *, model_ids: list[str], base_url: str, api_key: str, max_context_size: int
) -> None:
    config.providers[OPENAI_PROVIDER_KEY] = LLMProvider(
        type="openai_responses",
        base_url=base_url,
        api_key=SecretStr(api_key),
    )

    for key, model in list(config.models.items()):
        if model.provider == OPENAI_PROVIDER_KEY:
            del config.models[key]

    for model_id in model_ids:
        model_key = f"{OPENAI_MODEL_PREFIX}{model_id}"
        config.models[model_key] = LLMModel(
            provider=OPENAI_PROVIDER_KEY,
            model=model_id,
            max_context_size=max_context_size,
        )

    if not config.default_model and model_ids:
        config.default_model = f"{OPENAI_MODEL_PREFIX}{model_ids[0]}"


async def _sync_openai_models_if_configured(config: Config) -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    max_context_size = int(os.getenv("OPENAI_MODEL_MAX_CONTEXT_SIZE", "128000"))

    model_ids = await _fetch_openai_chat_models(base_url=base_url, api_key=api_key)
    if not model_ids:
        return False

    _upsert_openai_models(
        config,
        model_ids=model_ids,
        base_url=base_url,
        api_key=api_key,
        max_context_size=max_context_size,
    )

    config_for_save = load_config()
    _upsert_openai_models(
        config_for_save,
        model_ids=model_ids,
        base_url=base_url,
        api_key=api_key,
        max_context_size=max_context_size,
    )
    save_config(config_for_save)
    return True


@registry.command
async def model(app: Shell, args: str):
    """Switch LLM model or thinking mode"""
    from kimi_cli.llm import derive_model_capabilities

    soul = _ensure_kimi_soul(app)
    if soul is None:
        return
    config = soul.runtime.config

    if not config.is_from_default_location:
        console.print(
            "[yellow]Model switching requires the default config file; "
            "restart without --config/--config-file.[/yellow]"
        )
        return

    await refresh_managed_models(config)
    try:
        await _sync_openai_models_if_configured(config)
    except (aiohttp.ClientError, ValueError, OSError, ConfigError) as exc:
        console.print(f"[yellow]Failed to refresh OpenAI model list: {exc}[/yellow]")

    if not config.models:
        console.print(
            "[yellow]No models configured. "
            'Run "/login" or set OPENAI_API_KEY and retry /model.[/yellow]'
        )
        return

    # Find current model/thinking from runtime (may be overridden by --model/--thinking)
    curr_model_cfg = soul.runtime.llm.model_config if soul.runtime.llm else None
    curr_model_name: str | None = None
    if curr_model_cfg is not None:
        for name, model_cfg in config.models.items():
            if model_cfg == curr_model_cfg:
                curr_model_name = name
                break
    curr_thinking = soul.thinking

    # Step 1: Select model
    model_choices: list[tuple[str, str]] = []
    for name in sorted(config.models):
        model_cfg = config.models[name]
        provider_label = get_platform_name_for_provider(model_cfg.provider) or model_cfg.provider
        marker = " (current)" if name == curr_model_name else ""
        label = f"{model_cfg.model} ({provider_label}){marker}"
        model_choices.append((name, label))

    try:
        selected_model_name = await ChoiceInput(
            message="Select a model (↑↓ navigate, Enter select, Ctrl+C cancel):",
            options=model_choices,
            default=curr_model_name or model_choices[0][0],
        ).prompt_async()
    except (EOFError, KeyboardInterrupt):
        return

    if not selected_model_name:
        return

    selected_model_cfg = config.models[selected_model_name]
    selected_provider = config.providers.get(selected_model_cfg.provider)
    if selected_provider is None:
        console.print(f"[red]Provider not found: {selected_model_cfg.provider}[/red]")
        return

    # Step 2: Determine thinking mode
    capabilities = derive_model_capabilities(selected_model_cfg)
    new_thinking: bool

    if "always_thinking" in capabilities:
        new_thinking = True
    elif "thinking" in capabilities:
        thinking_choices: list[tuple[str, str]] = [
            ("off", "off" + (" (current)" if not curr_thinking else "")),
            ("on", "on" + (" (current)" if curr_thinking else "")),
        ]
        try:
            thinking_selection = await ChoiceInput(
                message="Enable thinking mode? (↑↓ navigate, Enter select, Ctrl+C cancel):",
                options=thinking_choices,
                default="on" if curr_thinking else "off",
            ).prompt_async()
        except (EOFError, KeyboardInterrupt):
            return

        if not thinking_selection:
            return

        new_thinking = thinking_selection == "on"
    else:
        new_thinking = False

    # Check if anything changed
    model_changed = curr_model_name != selected_model_name
    thinking_changed = curr_thinking != new_thinking

    if not model_changed and not thinking_changed:
        console.print(
            f"[yellow]Already using {selected_model_name} "
            f"with thinking {'on' if new_thinking else 'off'}.[/yellow]"
        )
        return

    # Save and reload
    prev_model = config.default_model
    prev_thinking = config.default_thinking
    config.default_model = selected_model_name
    config.default_thinking = new_thinking
    try:
        config_for_save = load_config()
        config_for_save.default_model = selected_model_name
        config_for_save.default_thinking = new_thinking
        save_config(config_for_save)
    except (ConfigError, OSError) as exc:
        config.default_model = prev_model
        config.default_thinking = prev_thinking
        console.print(f"[red]Failed to save config: {exc}[/red]")
        return

    console.print(
        f"[green]Switched to {selected_model_name} "
        f"with thinking {'on' if new_thinking else 'off'}. "
        "Reloading...[/green]"
    )
    raise Reload(session_id=soul.runtime.session.id)


@registry.command(aliases=["release-notes"])
@shell_mode_registry.command(aliases=["release-notes"])
def changelog(app: Shell, args: str):
    """Show release notes"""
    from rich.console import Group, RenderableType
    from rich.text import Text

    from kimi_cli.utils.rich.columns import BulletColumns

    renderables: list[RenderableType] = []
    for ver, entry in CHANGELOG.items():
        title = f"[bold]{ver}[/bold]"
        if entry.description:
            title += f": {entry.description}"

        lines: list[RenderableType] = [Text.from_markup(title)]
        for item in entry.entries:
            if item.lower().startswith("lib:"):
                continue
            lines.append(
                BulletColumns(
                    Text.from_markup(f"[grey50]{item}[/grey50]"),
                    bullet_style="grey50",
                ),
            )
        renderables.append(BulletColumns(Group(*lines)))

    with console.pager(styles=True):
        console.print(Group(*renderables))


@registry.command
@shell_mode_registry.command
def feedback(app: Shell, args: str):
    """Submit feedback to make Kimi Code CLI better"""
    import webbrowser

    ISSUE_URL = "https://github.com/MoonshotAI/kimi-cli/issues"
    if webbrowser.open(ISSUE_URL):
        return
    console.print(f"Please submit feedback at [underline]{ISSUE_URL}[/underline].")


@registry.command(aliases=["reset"])
async def clear(app: Shell, args: str):
    """Clear the context"""
    if _ensure_kimi_soul(app) is None:
        return
    await app.run_soul_command("/clear")
    raise Reload()


@registry.command(name="sessions", aliases=["resume"])
async def list_sessions(app: Shell, args: str):
    """List sessions and resume optionally"""
    soul = _ensure_kimi_soul(app)
    if soul is None:
        return

    work_dir = soul.runtime.session.work_dir
    current_session = soul.runtime.session
    current_session_id = current_session.id
    sessions = [
        session for session in await Session.list(work_dir) if session.id != current_session_id
    ]

    await current_session.refresh()
    sessions.insert(0, current_session)

    choices: list[tuple[str, str]] = []
    for session in sessions:
        time_str = format_relative_time(session.updated_at)
        marker = " (current)" if session.id == current_session_id else ""
        label = f"{session.title}, {time_str}{marker}"
        choices.append((session.id, label))

    try:
        selection = await ChoiceInput(
            message="Select a session to switch to (↑↓ navigate, Enter select, Ctrl+C cancel):",
            options=choices,
            default=choices[0][0],
        ).prompt_async()
    except (EOFError, KeyboardInterrupt):
        return

    if not selection:
        return

    if selection == current_session_id:
        console.print("[yellow]You are already in this session.[/yellow]")
        return

    console.print(f"[green]Switching to session {selection}...[/green]")
    raise Reload(session_id=selection)


@registry.command
def web(app: Shell, args: str):
    """Open Kimi Code Web UI in browser"""
    soul = _ensure_kimi_soul(app)
    session_id = soul.runtime.session.id if soul else None
    raise SwitchToWeb(session_id=session_id)


@registry.command
async def mcp(app: Shell, args: str):
    """Show MCP servers and tools"""
    from rich.console import Group, RenderableType
    from rich.text import Text

    from kimi_cli.soul.toolset import KimiToolset
    from kimi_cli.utils.rich.columns import BulletColumns

    soul = _ensure_kimi_soul(app)
    if soul is None:
        return
    toolset = soul.agent.toolset
    if not isinstance(toolset, KimiToolset):
        console.print("[red]KimiToolset required[/red]")
        return

    servers = toolset.mcp_servers

    if not servers:
        console.print("[yellow]No MCP servers configured.[/yellow]")
        return

    n_conn = sum(1 for s in servers.values() if s.status == "connected")
    n_tools = sum(len(s.tools) for s in servers.values())
    console.print(
        BulletColumns(
            Text.from_markup(
                f"[bold]MCP Servers:[/bold] {n_conn}/{len(servers)} connected, {n_tools} tools"
            )
        )
    )

    status_colors = {
        "connected": "green",
        "connecting": "cyan",
        "pending": "yellow",
        "failed": "red",
        "unauthorized": "red",
    }
    for name, info in servers.items():
        color = status_colors.get(info.status, "red")
        server_text = f"[{color}]{name}[/{color}]"
        if info.status == "unauthorized":
            server_text += " [grey50](unauthorized - run: kimi mcp auth {name})[/grey50]"
        elif info.status != "connected":
            server_text += f" [grey50]({info.status})[/grey50]"

        lines: list[RenderableType] = [Text.from_markup(server_text)]
        for tool in info.tools:
            lines.append(
                BulletColumns(
                    Text.from_markup(f"[grey50]{tool.name}[/grey50]"),
                    bullet_style="grey50",
                )
            )
        console.print(BulletColumns(Group(*lines), bullet_style=color))


from . import (  # noqa: E402
    debug,  # noqa: F401 # type: ignore[reportUnusedImport]
    oauth,  # noqa: F401 # type: ignore[reportUnusedImport]
    setup,  # noqa: F401 # type: ignore[reportUnusedImport]
    update,  # noqa: F401 # type: ignore[reportUnusedImport]
    usage,  # noqa: F401 # type: ignore[reportUnusedImport]
)
