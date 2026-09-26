"""A small Days at Three Branches starter built entirely from ``sandbox.village``."""

import math
from collections import deque

from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, day, geometry, layout, me, people, props

# Duration (in ticks) to hold "use" on a prop, by its transition type. Timed props use
# duration/4 so villagers don't stall for the full catalog duration (plot is shortened further,
# to 100); toggle/occupancy get a short hold. "sleep" is reserved for the end-of-day return
# home, not idle flavor.
_TIMED_USE_TICKS = {"plot": 100, "shrine": 75, "bell": 10, "pump": 3}
_TOGGLE_TYPES = ("lantern", "hearth", "stall")
_OCCUPANCY_TYPES = ("bench", "repair_bench")
_IDLE_EMOTES = tuple(emote for emote in action.EMOTES if emote != "sleep")
_TASK_TYPES = frozenset(
    {"plot", "shrine", "pump", "bell", "stall", "lantern", "hearth", "bench", "repair_bench", "board"}
)
# Each prop's untouched state. Anything else means another villager already handled it (a tended
# plot, a lit lantern) or is on it right now (an occupied bench), so there is no work left here.
_PROP_START_STATE = {
    "stall": "closed",
    "lantern": "unlit",
    "bench": "empty",
    "shrine": "untended",
    "plot": "overgrown",
    "hearth": "unlit",
    "repair_bench": "idle",
    "pump": "idle",
    "bell": "silent",
}
_STUCK_TICKS_LIMIT = 30  # abandon a destination after this many ticks of near-zero movement
_GREET_COOLDOWN_TICKS = 100  # per person, so a lingering neighbour isn't greeted every tick
_STARTLE_COOLDOWN_TICKS = 20  # separate & shorter, so startling doesn't use up the greet cooldown
_GREET_DELAY_TICKS = 10  # beat spent facing someone after noticing them, before actually greeting
_VISITOR_APPROACH_RANGE = 8.0  # blocks; only detour to greet if visitor is this close when noticed
_VISITOR_GREET_DISTANCE = 2.0  # blocks; walk to this distance before actually greeting
_BELL_RANGE = 50.0  # blocks; how far a villager will detour to answer a ringing bell
_RETURN_HOME_TICK = 1000  # of the 1200-tick day; when villagers head home for the "night"


def _cell_centre(cell: dict[str, int]) -> dict[str, float]:
    """Return the point at the centre of one village cell."""

    return {"x": cell["x"] + 0.5, "y": cell["y"] + 0.5}


def _prop_untouched(observation, prop) -> bool:
    """Whether a prop still needs doing and nobody else is on it.

    ``props.usable`` selects by reach and line of sight alone, regardless of facing, so it can
    hand back a prop that is not actually in the vision cone. ``props.state`` needs the cone, so
    that case reads as unknown here -- and an unconfirmed reading is treated as "leave it alone"
    rather than "assume untouched", since using an already-lit lantern would toggle it back off.
    """
    start = _PROP_START_STATE.get(prop["type"])
    if start is None:
        return True  # e.g. "board", which has no state to speak of
    return props.state(observation, prop["id"]) == start


def _nearest_walkable_cell(observation, cell, max_radius=15):
    """Return the closest walkable cell to one that may itself be blocked (e.g. a prop's footprint)."""
    if layout.walkable(observation, cell):
        return cell
    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidate = {"x": cell["x"] + dx, "y": cell["y"] + dy}
                if layout.walkable(observation, candidate):
                    return candidate
    return cell


def _bfs_path(observation, start_cell, dest_cell):
    """Shortest cardinal-step path of cells from start to dest, or None if unreachable."""
    if start_cell["x"] == dest_cell["x"] and start_cell["y"] == dest_cell["y"]:
        return []
    dest_key = (dest_cell["x"], dest_cell["y"])
    visited = {(start_cell["x"], start_cell["y"])}
    queue = deque([(start_cell, [])])
    while queue:
        current, path = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = {"x": current["x"] + dx, "y": current["y"] + dy}
            key = (neighbor["x"], neighbor["y"])
            if key in visited:
                continue
            if not layout.can_step(observation, current, neighbor):
                continue
            if key == dest_key:
                return path + [neighbor]
            visited.add(key)
            queue.append((neighbor, path + [neighbor]))
    return None


class Agent:
    """Leaves home, tends its plot, wanders the village, and reacts to people it sees."""

    def reset(self, seed: int, observation: ThreeBranchesObservation) -> None:
        """Initialize per-day state: home, plot, personality, and tracking structures."""

        # Home & doorway (static for the day)
        self.home = me.home(observation)
        self.home_doorway = layout.doorway(observation, self.home) if self.home != "none" else None

        # Claim closest plot to home
        here = me.position(observation)
        plots = [prop for prop in props.all(observation) if prop["type"] == "plot"]
        self.claimed_plot = min(
            plots,
            key=lambda p: geometry.distance(here, _cell_centre(p["cell"])),
        ) if plots else None

        # Personality: which destination-picking strategy this villager follows all day
        rng = me.rng(observation, seed)
        self.personality = rng.choice(["wander_random", "visit_building", "seek_tasks"])
        self.rng = rng  # kept for destination picks and random emotes during the day

        # Day-phase tracking
        self.phase = "plot"  # "plot" -> "wander" -> "returning"
        self.used_props = set()  # prop ids already used today, never repeated
        self.greeted = {}  # {player_id: last_greeted_tick}, per-person greet cooldown
        self.startled = {}  # {player_id: last_startled_tick}, separate from greeted on purpose
        self.pending_greet = None  # {"id", "ready_tick", "approach_player"?} while greeting

        # Destination & task state
        self.current_destination = None  # position dict to walk toward
        self.active_task = None  # {"prop_id": str, "ticks_remaining": int} while mid-use
        self.use_counter = 0

        # Pathfinding cache
        self._path = None  # cached cardinal-step path to current destination
        self._path_destination = None  # (x, y) tuple of cached path's target
        self.stuck_ticks = 0  # consecutive near-zero-movement ticks while walking a destination

        # Greeting handed from act() to chat(), which the harness calls later the same tick
        self._pending_chat = None

    def _prop_use_ticks(self, prop_type: str) -> int:
        """How many ticks to hold "use" on a prop, based on its catalog transition type."""
        if prop_type in _TIMED_USE_TICKS:
            return _TIMED_USE_TICKS[prop_type]
        if prop_type in _TOGGLE_TYPES:
            return self.rng.randint(5, 8)
        if prop_type in _OCCUPANCY_TYPES:
            return self.rng.randint(5, 8)
        return 1  # e.g. "board": no state to hold, just a passing glance

    def _walk_toward(self, destination: dict[str, float], observation: ThreeBranchesObservation) -> float:
        """Heading toward destination, following a cached path around obstacles."""
        here = me.position(observation)
        here_cell = layout.cell_at(observation, here)
        dest_cell = layout.cell_at(observation, destination)
        if here_cell is None or dest_cell is None:
            return geometry.heading_to(here, destination)
        dest_cell = _nearest_walkable_cell(observation, dest_cell)

        cache_key = (dest_cell["x"], dest_cell["y"])
        if self._path_destination != cache_key or self._path is None:
            self._path = _bfs_path(observation, here_cell, dest_cell) or []
            self._path_destination = cache_key

        while self._path and self._path[0]["x"] == here_cell["x"] and self._path[0]["y"] == here_cell["y"]:
            self._path.pop(0)

        if not self._path:
            return geometry.heading_to(here, destination)
        return geometry.heading_to(here, _cell_centre(self._path[0]))

    def _pick_destination(self, observation: ThreeBranchesObservation) -> dict[str, float]:
        """Pick the next place to head, per this villager's personality. Always returns an
        actually-walkable point, since a prop or building cell can itself be blocked by
        collision (a raw prop cell led to villagers oscillating against it)."""
        here = me.position(observation)

        if self.personality == "wander_random":
            for _ in range(20):
                angle = math.radians(self.rng.uniform(0.0, 360.0))
                reach = self.rng.uniform(5.0, 30.0)
                point = {"x": here["x"] + reach * math.cos(angle), "y": here["y"] + reach * math.sin(angle)}
                cell = layout.cell_at(observation, point)
                if cell is not None and layout.walkable(observation, cell):
                    return _cell_centre(cell)
            return here

        if self.personality == "visit_building":
            candidates = list(layout.buildings(observation))
            self.rng.shuffle(candidates)
            for building in candidates:
                doorway = layout.doorway(observation, building["id"])
                if doorway is not None:
                    return doorway
            return here

        # "seek_tasks"
        candidates = [
            prop for prop in props.all(observation)
            if prop["type"] in _TASK_TYPES and prop["id"] not in self.used_props
        ]
        if candidates:
            cell = self.rng.choice(candidates)["cell"]
            return _cell_centre(_nearest_walkable_cell(observation, cell))
        return here

    def _bell_destination(self, observation: ThreeBranchesObservation) -> dict[str, float] | None:
        """If the bell is ringing and within range, head there; otherwise None."""
        if not day.bell_ringing(observation):
            return None
        here = me.position(observation)
        bell = next((prop for prop in props.all(observation) if prop["type"] == "bell"), None)
        if bell is None:
            return None
        bell_pos = _cell_centre(bell["cell"])
        if geometry.distance(here, bell_pos) > _BELL_RANGE:
            return None
        return bell_pos

    def _react_to_people(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction | None:
        """Stop and turn toward whoever just showed up, or None when nobody is due a reaction.

        Reactions outrank every task, so this runs before the phase logic. Pausing mid-use costs
        nothing: the use counter simply doesn't advance this tick and resumes on the next one.
        """
        here = me.position(observation)
        tick = day.tick(observation)
        seen = people.seen(observation)
        nearby = people.nearby(observation)
        seen_ids = {person["id"] for person in seen}
        # Chat is only delivered to someone in hearing range, so speaking needs this, not sight
        nearby_ids = {person["id"] for person in nearby}

        def find(player_id: str):
            """That person's current record, from sight or hearing, or None once they are gone."""
            for person in seen:
                if person["id"] == player_id:
                    return person
            for person in nearby:
                if person["id"] == player_id:
                    return person
            return None

        # Already noticed someone: walk to them if they started close by, then greet
        if self.pending_greet is not None:
            person = find(self.pending_greet["id"])
            if person is None:
                self.pending_greet = None  # they left before we could say anything
            else:
                facing = geometry.heading_to(here, person["position"])
                dist_to_person = geometry.distance(here, person["position"])

                # If the visitor started within approach range, walk over before greeting
                if self.pending_greet.get("approach_player") and dist_to_person > _VISITOR_GREET_DISTANCE:
                    return action.walk(self._walk_toward(person["position"], observation), 1.0, "none")

                if tick < self.pending_greet["ready_tick"]:
                    return action.stand(facing, "none")

                player_id = person["id"]
                self.greeted[player_id] = tick
                self.pending_greet = None
                if people.is_visitor(player_id):
                    if player_id in nearby_ids:  # close enough that a greeting would carry
                        self._pending_chat = {"to": player_id, "text": "Hello."}
                    return action.stand(facing, "wave")
                return action.stand(facing, "nod")

        def greet_due(player_id: str) -> bool:
            return tick - self.greeted.get(player_id, -_GREET_COOLDOWN_TICKS) >= _GREET_COOLDOWN_TICKS

        def startle_due(player_id: str) -> bool:
            return tick - self.startled.get(player_id, -_STARTLE_COOLDOWN_TICKS) >= _STARTLE_COOLDOWN_TICKS

        def notice(person, expression: str):
            """Turn toward someone and start the beat that ends in a wave or a nod."""
            approach = (
                people.is_visitor(person["id"])
                and geometry.distance(here, person["position"]) <= _VISITOR_APPROACH_RANGE
            )
            self.pending_greet = {
                "id": person["id"],
                "ready_tick": tick + _GREET_DELAY_TICKS,
                "approach_player": approach,
            }
            return action.stand(geometry.heading_to(here, person["position"]), expression)

        # The visitor comes first
        visitor = next((person for person in seen if people.is_visitor(person["id"])), None)
        if visitor is not None and greet_due(visitor["id"]):
            return notice(visitor, "none")

        for person in seen:
            if not people.is_visitor(person["id"]) and greet_due(person["id"]):
                return notice(person, "none")

        # Heard but not seen means they are in the blind spot behind us, so the beat opens with a
        # startle instead. It doesn't use up the greet cooldown, so the wave or nod still follows.
        for person in nearby:
            if person["id"] in seen_ids:
                continue
            if startle_due(person["id"]) and greet_due(person["id"]):
                self.startled[person["id"]] = tick
                return notice(person, "startle")

        return None

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Choose one action from current observation and persistent day state."""

        here = me.position(observation)
        heading = me.heading(observation)

        reaction = self._react_to_people(observation)
        if reaction is not None:
            return reaction

        if self.phase != "returning" and day.tick(observation) >= _RETURN_HOME_TICK:
            # Time to head home, whatever else was in progress
            self.phase = "returning"
            self.active_task = None
            self.current_destination = None
            self._path = None
            self.stuck_ticks = 0

        # Phase: plot (tend the home plot, then switch to wander)
        if self.phase == "plot":
            if self.active_task is None:
                # Initialize plot task
                self.active_task = {
                    "prop_id": self.claimed_plot["id"],
                    "ticks_remaining": self._prop_use_ticks(self.claimed_plot["type"]),
                }
                self.use_counter = 0

            if self.use_counter >= self.active_task["ticks_remaining"]:
                # Done tending plot
                self.used_props.add(self.claimed_plot["id"])
                self.phase = "wander"
                self.active_task = None
                self.use_counter = 0
            else:
                # Walk to plot or use it if close enough
                plot_pos = _cell_centre(self.claimed_plot["cell"])
                dist_to_plot = geometry.distance(here, plot_pos)

                if dist_to_plot < 1.5:  # Close enough to use (prop_reach is 1.5)
                    self.use_counter += 1
                    return action.stand(heading, "use")
                else:
                    return action.walk(self._walk_toward(plot_pos, observation), 1.0, "none")

        # Phase: wander (walk to a personality-driven destination, using anything usable
        # spotted along the way, sweeping at the end of each leg, then picking a new one)
        if self.phase == "wander":
            if self.active_task is None:
                usable = props.usable(observation)
                if (
                    usable is not None
                    and usable["id"] not in self.used_props
                    and _prop_untouched(observation, usable)
                ):
                    self.active_task = {
                        "prop_id": usable["id"],
                        "ticks_remaining": self._prop_use_ticks(usable["type"]),
                    }
                    self.use_counter = 0

            if self.active_task is not None:
                if self.use_counter >= self.active_task["ticks_remaining"]:
                    self.used_props.add(self.active_task["prop_id"])
                    self.active_task = None
                    self.use_counter = 0
                else:
                    self.use_counter += 1
                    return action.stand(heading, "use")

            bell_destination = self._bell_destination(observation)
            if bell_destination is not None and self.current_destination != bell_destination:
                self.current_destination = bell_destination
                self._path = None
                self.stuck_ticks = 0

            if self.current_destination is None:
                # Deciding beat: pick where to head next and show it with a random emote
                self.current_destination = self._pick_destination(observation)
                self.stuck_ticks = 0
                self._path = None  # force a fresh path for the new destination
                return action.stand(heading, self.rng.choice(_IDLE_EMOTES))

            if geometry.distance(here, self.current_destination) < 2.0:
                # Arrived: sweep, then pick a fresh destination next tick
                self.current_destination = None
                return action.stand(heading, "sweep")

            # Give up on a destination that isn't actually reachable instead of circling on it
            if me.moved(observation) < 0.05:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = 0
            if self.stuck_ticks >= _STUCK_TICKS_LIMIT:
                self.current_destination = None
                self.stuck_ticks = 0
                self._path = None
                return action.stand(heading, "shrug")

            return action.walk(self._walk_toward(self.current_destination, observation), 1.0, "none")

        # Phase: returning (walk home for the night, then sleep for the rest of the day)
        if self.phase == "returning":
            if self.home_doorway is None:  # no home to return to (e.g. the visitor seat)
                return action.stand(heading, "sleep")
            here_cell = layout.cell_at(observation, here)
            ground = layout.ground_at(observation, here_cell) if here_cell is not None else None
            if ground == "interior" or geometry.distance(here, self.home_doorway) < 1.0:
                return action.stand(heading, "sleep")
            return action.walk(self._walk_toward(self.home_doorway, observation), 1.0, "none")

        return action.stand(heading, "none")

    # Optional: messaging. On your turn, chat receives messages addressed to your player since
    # its previous turn. Return messages with a recipient and text, or nothing to stay silent.
    # Use None as the recipient to broadcast. Every message is recorded and shown in replays.
    #
    def chat(self, inbox: list[dict]) -> list[dict] | None:
        """Send the greeting act() queued when it waved at the visitor."""
        if self._pending_chat is None:
            return None
        message = self._pending_chat
        self._pending_chat = None
        return [message]