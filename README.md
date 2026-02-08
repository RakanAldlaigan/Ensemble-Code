EnsembleCode

EnsembleCode is a terminal UI (TUI) coding assistant built around a multi-agent architecture.
What It Does

    Runs a main Super-agent that coordinates work.
    Delegates specialized tasks to subagents when needed.
    Provides an interactive TUI workflow for agent-driven development.

Current Status

    The current TUI codebase is forked from Kimi CLI.
    EnsembleCode-specific behavior and orchestration are being developed on top of that fork.

Project Goal

Create a practical multi-agent coding experience where a central orchestrator can route tasks to the best specialized agent and combine their outputs efficiently.


Running the project

Install dependencies or use github codespace 

export openai api key to use their models
>export OPENAI_API_KEY=''
start the 'super model' (make sure uv is installed)
>uv run kimi
Chose model (GPT-5 nano recomended)
>/model 

start prompting!

Example prompt:
user:
what tools can you use?
super agent:
• Task (functions.Task) — spawn a focused subagent (e.g., "coder") to perform a narrow task.
• SetTodoList (functions.SetTodoList) — create/update a todo list for multi-step tasks.
• Shell (functions.Shell) — run bash commands (ls, cat, grep, curl, etc.) inside the working directory.
• ReadFile (functions.ReadFile) — read text files with line numbers.
• Glob (functions.Glob) — find files/directories by glob patterns.
• Grep (functions.Grep) — fast content search (ripgrep-style) across files.
• WriteFile (functions.WriteFile) — write or overwrite text files.
• StrReplaceFile (functions.StrReplaceFile) — perform targeted string replacements inside files.
• FetchURL (functions.FetchURL) — fetch a web page and extract main text.
• ExecuteCalculator (functions.ExecuteCalculator) — run calculator-focused numerical evaluations.
• ExecuteCode (functions.ExecuteCode) — generate and run Python code in a sandboxed notebook.
• ExecutePDFExtractions (functions.ExecutePDFExtractions) — structured extraction from PDF content.
• SleepTool (functions.SleepTool) — wait/sleep (mainly for testing).
• multi_tool_use.parallel — wrapper to run multiple compatible tools in parallel.

Note, the following tools are LLMs that are have the ability to excute code in a secure sandbox.
• ExecuteCalculator (functions.ExecuteCalculator) — run calculator-focused numerical evaluations.
• ExecuteCode (functions.ExecuteCode) — generate and run Python code in a sandboxed notebook.
• ExecutePDFExtractions (functions.ExecutePDFExtractions) — structured extraction from PDF content.

If you want to see the agent to use a tool, instruct it explicity