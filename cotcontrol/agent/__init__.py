"""Minimal sandboxed tool-use agent loop (virtual filesystem, text tool protocol).

    protocol.py   <tool_call>{json}</tool_call> protocol + forgiving parser, system prompt
    loop.py       VirtualFS tools, Episode/Turn records, lockstep batched driver
"""

from cotcontrol.agent.loop import (  # noqa: F401
    DEFAULT_PREAMBLE,
    TOOL_NAMES,
    TOOLS,
    Episode,
    EpisodeSpec,
    Turn,
    VirtualFS,
    run_episodes,
)
from cotcontrol.agent.protocol import (  # noqa: F401
    ToolCall,
    ToolSpec,
    parse_tool_calls,
    render_system_prompt,
)
