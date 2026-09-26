"""A small Days at Three Branches starter built entirely from ``sandbox.village``."""

import math
from collections import deque

from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, day, geometry, layout, me, people, props

# Duration (in ticks) to hold "use" on a prop, by its transition type. Timed props use
# duration/4 so villagers don't stall for the full catalog duration (plot is shortened further,
# to 50, since only the two farmers ever tend one); toggle/occupancy get a short hold. "sleep" is
# reserved for the end-of-day return home, not idle flavor.
_TIMED_USE_TICKS = {"plot": 50, "shrine": 75, "bell": 10, "pump": 3}
_TOGGLE_TYPES = ("lantern", "hearth", "stall")
_OCCUPANCY_TYPES = ("bench", "repair_bench")
_IDLE_EMOTES = tuple(emote for emote in action.EMOTES if emote != "sleep")
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

_HOP_REACH_MIN = 5.0  # blocks; short legs toward a destination, sweeping between each
_HOP_REACH_MAX = 30.0
_OCCUPIED_CHECK_RANGE = 2.0  # blocks; someone this close to a prop already has it, pick another

_BREAK_MIN_TICKS = 15  # after completing a task, idle-hop for a random duration before next task
_BREAK_MAX_TICKS = 45

_LIGHT_TYPES = ("lantern", "hearth")

# Which prop types each role deliberately seeks out, once no one else is already on the nearest
# one. Only farmers touch plots; the building-wanderer role has no props of its own (see
# _advance_building) so it is intentionally left out of this mapping.
_ROLE_TARGET_TYPES = {
    "farmer": ("plot",),
    "shrine_keeper": ("shrine",),
    "wellkeeper": ("pump",),
    "stall_tender": ("stall",),
    "occupancy_tender": ("bench", "repair_bench"),
    "lighter": _LIGHT_TYPES,
}

# 10 seats this season: 2 farmers, 2 shrine keepers, 1 wellkeeper, 2 stall tenders, 1 occupancy
# tender, 1 lighter, 1 building-wanderer. Indexed by player number (wraps if the cast is smaller).
_ROLE_ORDER = (
    "farmer",
    "farmer",
    "shrine_keeper",
    "shrine_keeper",
    "wellkeeper",
    "stall_tender",
    "stall_tender",
    "occupancy_tender",
    "lighter",
    "building_wanderer",
)


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
    """Leaves home, does its role's rounds, wanders the village, and reacts to people it sees."""

    def reset(self, seed: int, observation: ThreeBranchesObservation) -> None:
        """Initialize per-day state: home, role, and tracking structures."""

        # Home & doorway (static for the day)
        self.home = me.home(observation)
        self.home_doorway = layout.doorway(observation, self.home) if self.home != "none" else None

        # Role: fixed by player number, not randomized. Determines which props (if any) this
        # villager deliberately seeks out all day; see _ROLE_TARGET_TYPES and _advance_building.
        index = int(me.player_id(observation).split("_")[1]) - 1
        self.role = _ROLE_ORDER[index % len(_ROLE_ORDER)]

        self.rng = me.rng(observation, seed)  # kept for hop picks, idle emotes, and building picks

        # Day-phase tracking
        self.phase = "task"  # "task" -> "returning"
        self.announced_role = False  # broadcast role once at start of day
        self.used_props = set()  # prop ids already used today, never repeated
        self.greeted = {}  # {player_id: last_greeted_tick}, per-person greet cooldown
        self.startled = {}  # {player_id: last_startled_tick}, separate from greeted on purpose
        self.pending_greet = None  # {"id", "ready_tick", "approach_player"?} while greeting

        # Role-task state: the prop (or, for building_wanderer, doorway) currently claimed as
        # this villager's next job, and the short waypoint it is hopping toward on the way there
        self.role_target = None  # {"prop_id", "type", "position"} or None
        self.hop_point = None
        self.active_task = None  # {"prop_id": str, "ticks_remaining": int} while mid-use
        self.use_counter = 0
        self.break_ticks_remaining = 0  # idle-hop after completing a task before picking next

        # Direct-walk destination, used only for the bell response (no hops -- answer it quickly)
        self.current_destination = None

        # Pathfinding cache, shared by both the bell response and the hop travel above
        self._path = None  # cached cardinal-step path to whichever destination is active
        self._path_destination = None  # (x, y) tuple of cached path's target
        self.stuck_ticks = 0  # consecutive near-zero-movement ticks while walking a destination

        # Greeting handed from act() to chat(), which the harness calls later the same tick
        self._pending_chat = None

    def _prop_use_ticks(self, prop_type: str) -> int:
        """How many ticks to hold "use" on a prop, based on its catalog transition type."""
        if prop_type in _TIMED_USE_TICKS:
            return _TIMED_USE_TICKS[prop_type]
        if prop_type in _TOGGLE_TYPES:
            return 3
        if prop_type in _OCCUPANCY_TYPES:
            return self.rng.randint(4, 6)
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

    def _random_nearby_point(self, observation: ThreeBranchesObservation, here) -> dict[str, float]:
        """A random walkable point a short hop away, for idling when a role has no work left."""
        for _ in range(20):
            angle = math.radians(self.rng.uniform(0.0, 360.0))
            reach = self.rng.uniform(_HOP_REACH_MIN, _HOP_REACH_MAX)
            point = {"x": here["x"] + reach * math.cos(angle), "y": here["y"] + reach * math.sin(angle)}
            cell = layout.cell_at(observation, point)
            if cell is not None and layout.walkable(observation, cell):
                return _cell_centre(cell)
        return here

    def _pick_hop_point(self, observation: ThreeBranchesObservation, here, target_pos) -> dict[str, float]:
        """A short waypoint along the real path toward target_pos, so travel comes in small hops."""
        here_cell = layout.cell_at(observation, here)
        dest_cell = layout.cell_at(observation, target_pos)
        if here_cell is None or dest_cell is None:
            return target_pos
        dest_cell = _nearest_walkable_cell(observation, dest_cell)
        path = _bfs_path(observation, here_cell, dest_cell) or []
        if not path:
            return target_pos
        hop_len = max(1, int(self.rng.uniform(_HOP_REACH_MIN, _HOP_REACH_MAX)))
        index = min(hop_len, len(path)) - 1
        return _cell_centre(path[index])

    def _available_props(self, observation: ThreeBranchesObservation, prop_types):
        """All untended props of these types that nobody visible is already working on."""
        watchers = [person["position"] for person in people.seen(observation)]
        watchers += [person["position"] for person in people.nearby(observation)]
        candidates = []
        for prop in props.all(observation):
            if prop["type"] not in prop_types:
                continue
            if prop["id"] in self.used_props:
                continue
            if not _prop_untouched(observation, prop):
                continue
            prop_pos = _cell_centre(prop["cell"])
            if any(geometry.distance(prop_pos, watcher) <= _OCCUPIED_CHECK_RANGE for watcher in watchers):
                continue  # someone is already standing right on this one -- let them have it
            candidates.append(prop)
        return candidates

    def _nearest_available_prop(self, observation: ThreeBranchesObservation, prop_types):
        """Closest untended prop of these types that nobody visible is already working, or None."""
        here = me.position(observation)
        candidates = self._available_props(observation, prop_types)
        if not candidates:
            return None
        return min(candidates, key=lambda prop: geometry.distance(here, _cell_centre(prop["cell"])))

    def _random_available_prop(self, observation: ThreeBranchesObservation, prop_types):
        """Random untended prop of these types that nobody visible is already working, or None."""
        candidates = self._available_props(observation, prop_types)
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _opportunistic_light(self, observation: ThreeBranchesObservation):
        """An unlit lantern/hearth right here to toggle on the way, regardless of role."""
        usable = props.usable(observation)
        if usable is None or usable["type"] not in _LIGHT_TYPES:
            return None
        if usable["id"] in self.used_props:
            return None
        if not _prop_untouched(observation, usable):
            return None
        return usable

    def _advance_hop(self, observation: ThreeBranchesObservation, heading: float, here, target_pos, sweep_at_target=False):
        """Walk toward target_pos in short hops, sweeping at each waypoint before the next leg."""
        if self.hop_point is None:
            self.hop_point = self._pick_hop_point(observation, here, target_pos)
            self._path = None
            self.stuck_ticks = 0

        # On the final leg the caller decides arrival (prop reach is tighter than 2.0), so don't
        # sweep-stop short of the target -- that looped forever just outside use range.
        final_leg = geometry.distance(self.hop_point, target_pos) < 1.0
        if (sweep_at_target or not final_leg) and geometry.distance(here, self.hop_point) < 2.0:
            self.hop_point = None
            return action.stand(heading, "sweep")

        if me.moved(observation) < 0.05:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        if self.stuck_ticks >= _STUCK_TICKS_LIMIT:
            self.hop_point = None
            self.stuck_ticks = 0
            self._path = None
            return action.stand(heading, "shrug")

        return action.walk(self._walk_toward(self.hop_point, observation), 1.0, "none")

    def _idle_hop(self, observation: ThreeBranchesObservation, heading: float, here):
        """Keep moving in short hops when a role has run out of things to do for the day."""
        if self.hop_point is None:
            self.hop_point = self._random_nearby_point(observation, here)
            self._path = None
            self.stuck_ticks = 0
            return action.stand(heading, self.rng.choice(_IDLE_EMOTES))
        return self._advance_hop(observation, heading, here, self.hop_point, sweep_at_target=True)

    def _advance_building(self, observation: ThreeBranchesObservation, heading: float, here):
        """The building-wanderer role: head to a random building's doorway, sweep, pick another."""
        if self.role_target is None:
            candidates = list(layout.buildings(observation))
            self.rng.shuffle(candidates)
            doorway = None
            for building in candidates:
                doorway = layout.doorway(observation, building["id"])
                if doorway is not None:
                    break
            if doorway is None:
                return self._idle_hop(observation, heading, here)
            self.role_target = {"prop_id": None, "type": None, "position": doorway}
            self.hop_point = None

        target_pos = self.role_target["position"]
        if geometry.distance(here, target_pos) <= 2.0:
            self.role_target = None
            self.hop_point = None
            return action.stand(heading, "sweep")

        return self._advance_hop(observation, heading, here, target_pos)

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

    def _task_tick(self, observation: ThreeBranchesObservation, heading: float, here) -> ThreeBranchesAction:
        """One tick of role-driven work: answer the bell, grab a passing light, else do the job."""

        # The bell outranks the day's role work, and is answered directly (no hops -- be quick)
        bell_destination = self._bell_destination(observation)
        if bell_destination is not None:
            if self.current_destination != bell_destination:
                self.current_destination = bell_destination
                self._path = None
                self.stuck_ticks = 0
            if geometry.distance(here, self.current_destination) < 2.0:
                self.current_destination = None
                return action.stand(heading, "sweep")
            return action.walk(self._walk_toward(self.current_destination, observation), 1.0, "none")

        # Everyone, whatever their role, lights any unlit lantern/hearth they pass close enough to use
        if self.active_task is None:
            light = self._opportunistic_light(observation)
            if light is not None:
                self.active_task = {"prop_id": light["id"], "ticks_remaining": self._prop_use_ticks(light["type"])}
                self.use_counter = 0
            elif self.role_target is not None and self.role_target["prop_id"] is not None:
                if geometry.distance(here, self.role_target["position"]) <= geometry.PROP_REACH:
                    self.active_task = {
                        "prop_id": self.role_target["prop_id"],
                        "ticks_remaining": self._prop_use_ticks(self.role_target["type"]),
                    }
                    self.use_counter = 0

        if self.active_task is not None:
            if self.use_counter >= self.active_task["ticks_remaining"]:
                self.used_props.add(self.active_task["prop_id"])
                if self.role_target is not None and self.role_target["prop_id"] == self.active_task["prop_id"]:
                    self.role_target = None
                    self.hop_point = None
                    # Start a break after completing a task
                    self.break_ticks_remaining = self.rng.randint(_BREAK_MIN_TICKS, _BREAK_MAX_TICKS)
                self.active_task = None
                self.use_counter = 0
            else:
                self.use_counter += 1
                return action.stand(heading, "use")

        # Break after completing a task: idle-hop for a random duration (still lighting lamps en route)
        if self.break_ticks_remaining > 0:
            self.break_ticks_remaining -= 1
            return self._idle_hop(observation, heading, here)

        if self.role == "building_wanderer":
            return self._advance_building(observation, heading, here)

        if self.role_target is None:
            prop = self._random_available_prop(observation, _ROLE_TARGET_TYPES[self.role])
            if prop is None:
                return self._idle_hop(observation, heading, here)  # nothing left today; keep it lively
            self.role_target = {
                "prop_id": prop["id"],
                "type": prop["type"],
                "position": _cell_centre(_nearest_walkable_cell(observation, prop["cell"])),
            }
            self.hop_point = None

        if geometry.distance(here, self.role_target["position"]) <= geometry.PROP_REACH:
            return action.stand(heading, "none")  # arrived; active_task starts from here next tick

        return self._advance_hop(observation, heading, here, self.role_target["position"])

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Choose one action from current observation and persistent day state."""

        here = me.position(observation)
        heading = me.heading(observation)

        # Announce role once at start of day
        if not self.announced_role:
            self.announced_role = True
            self._pending_chat = {"to": None, "text": f"I'm a {self.role.replace('_', ' ')}."}
            return action.stand(heading, "none")

        reaction = self._react_to_people(observation)
        if reaction is not None:
            return reaction

        if self.phase != "returning" and day.tick(observation) >= _RETURN_HOME_TICK:
            # Time to head home, whatever else was in progress
            self.phase = "returning"
            self.active_task = None
            self.role_target = None
            self.hop_point = None
            self.current_destination = None
            self._path = None
            self.stuck_ticks = 0

        if self.phase == "task":
            return self._task_tick(observation, heading, here)

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
