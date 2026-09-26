# Days at Three Branches agent

Edit `agent.py` to give every one of your villagers a routine. The platform runs a separate `Agent` instance for each NPC, so instances do not share variables or memory. The supplied `sandbox/` directory contains the local runner, types, and village helpers. Leave it unchanged, and leave `requirements.in` and `requirements.txt` alone: the pinned packages match the server.

Start with the [Getting Started guide](https://vox-deorum.github.io/game-sandbox/students/getting-started/). Then run these commands from this folder:

```console
python -m sandbox watch  # watch your villagers and the scripted visitor
python -m sandbox test   # run the checks
python -m sandbox eval   # run repeatable automated days
```

## Files you will use

| Path | Purpose |
| --- | --- |
| `agent.py` | Your `Agent` implementation and starter TODOs. |
| `environment.md` | Rules, helpers, observations, and local commands. |
| `manifest.json` | Tells Game Sandbox where the agent class lives. |
| `season.json` | Optional local season settings downloaded from My Submissions. |
| `tests/` | Checks your submission should pass. |
| `sandbox/` | Local game, helpers, and types. Do not edit it. |
| `requirements.txt` | Exact Python package versions used by the server. |
| `requirements-dev.txt` | Test dependencies. |
| `.env.example` | Example local LLM settings. |

The starter shows `action.walk`, `action.stand`, an emote, and `use`. Read [`environment.md`](environment.md) before changing it. Begin with one behavior that you can recognize in `watch`, then make it more responsive to the people and props it sees.

Chat is optional. If you add `chat(self, inbox)` to your agent, send and receive raw message dictionaries that use canonical player IDs such as `"player_0"` and `"player_1"`. For example, return `[{"to": "player_0", "text": "Hello."}]` to send a direct message. The village helpers do not include a chat namespace. Read [`environment.md`](environment.md#chat-with-other-agents) for the inbox format, broadcasts, and delivery timing.

In `watch` and `eval`, your `Agent` controls the whole cast, and `scripted_visitor` controls the visitor. The `naive` built-in is the simple baseline for the cast. When you are ready to submit, follow the [submitting guide](https://vox-deorum.github.io/game-sandbox/students/submitting/).

## Optional LLM API

If your instructor enables model calls, follow [Using the LLM API](llm.md). Copy `.env.example` to `.env`, add the endpoint and key, and never commit either secret.

Test the connection with:

```console
python -m sandbox llm
```

## Design Goal
Season 1: The goal of this season was to have the characters move around, perform tasks, and live a lively village life while interacting with the player. The characters now behave according to their roles, going around wandering randomly while also making their way to assigned tasks and greeting each other.

## Reflection
Season 2: I worked through and tested different roles of different amounts for the characters. I also tried to tie that a little more to pathfinding, where they wander but in an intentional direction to their next objective. I also had them check whether objects were occupied, and the logic for interactions with players was finalised. I tried to get them to cover most visible tasks. However, with the limited number of certain objects or certain players, it seemed that some players did the same task over and over. Still, the players seemed to interact with visible tasks more and group in an area where the player can see interactions more clearly.

## AI Use Disclosure and Reflection
I used Claude Code in VS Code. It coded all of this based on my pseudocode and instructions. I verified each set of code it wrote before approving it, reverted and asked for changes as needed, tested by watching and eval, and discussed my ideas for solutions before committing to one. I did not write any code directly, but read through it to check it. I had it set to ask for approval before committing anything, and that worked especially during testing and understanding what code was being changed. Planning mode helped a lot, but it's hard to keep it in check as to not add features without asking me. 

