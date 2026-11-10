"""Stateless semantic RAM readers for Seaquest, Frostbite, Pong and Boxing.

Coordinates are TOP-LEFT bounding boxes in the uncropped ALE 160 x 210 screen.
Only numpy is needed; no OCAtari install, emulator, model or GPU is required.
Mapping references (consulted 2026-11-10):
https://github.com/k4ntz/OC_Atari/tree/master/ocatari/ram
See RAM_VERIFICATION.md for scope, byte meanings and limitations.
"""

import numpy as np


GAMES = ("seaquest", "frostbite", "pong", "boxing")
ALE_NAMES = dict(zip(GAMES, ("Seaquest", "Frostbite", "Pong", "Boxing")))
SOURCE_URLS = {
    game: f"https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/{game}.py"
    for game in GAMES
}


def canonical_game(game):
    name = game.lower().removeprefix("ale/").split("-")[0]
    if name not in GAMES:
        raise ValueError(f"Unsupported game {game!r}; expected one of {GAMES}")
    return name


def validate_ram(ram):
    """Convert BEFORE arithmetic to avoid uint8 overflow/underflow."""
    ram = np.asarray(ram)
    if ram.shape != (128,) or not np.issubdtype(ram.dtype, np.integer):
        raise ValueError("RAM must be an integer array with shape (128,)")
    if np.any((ram < 0) | (ram > 255)):
        raise ValueError("RAM byte values must be in [0, 255]")
    if np.all(ram == ram[0]):
        raise ValueError("All 128 RAM bytes are identical. This can be the old ALE/NumPy "
                         "getRAM() binding bug; collect again using read_ale_ram(ale).")
    return ram.astype(np.int64)


def read_ale_ram(ale):
    """Avoid the no-argument getRAM binding bug in ALE 0.8.1 + NumPy 2.

    Newer ALE versions may only support the no-argument overload. Always copy,
    validate shape/range, and reject an implausible constant RAM snapshot.
    """
    buffer = np.empty(128, dtype=np.uint8)
    try:
        ale.getRAM(buffer)
    except TypeError:
        buffer = np.array(ale.getRAM(), copy=True)
    validate_ram(buffer)
    return buffer


def _object(objects, ram, name, kind, box, addresses, **attributes):
    x, y, w, h = map(int, box)
    # A negative coordinate may be legitimate for a partially visible object.
    if w > 0 and h > 0 and x < 160 and x + w > 0 and y < 210 and y + h > 0:
        objects.append(dict(id=name, kind=kind, xywh=[x, y, w, h],
                            ram={str(a): int(ram[a]) for a in addresses},
                            attributes=attributes))


def _bcd_or_none(value):
    high, low = int(value) >> 4, int(value) & 15
    return 10 * high + low if high < 10 and low < 10 else None


def _seaquest(r, objects, warnings):
    scalars = dict(collected_divers=int(r[62]), oxygen=int(r[102]),
                   reserve_lives=int(r[59]), level_raw=int(r[61]),
                   facing_left=int(r[86] != 0), death_animation_raw=int(r[105]))
    if r[62] > 6:
        warnings.append("Collected-diver byte is outside the expected 0..6 range")
    if r[102] > 64:
        warnings.append("Oxygen byte is outside the expected 0..64 range")
    if not 0 < r[105] < 15:
        _object(objects, r, "player", "player", (r[70], r[97] + 33, 16, 11),
                [70, 97, 86, 105])
    for lane in range(4):  # lane 0 = bottom; lane 3 = top
        submarine = 3 < r[89 + lane] % 8 < 7
        x_addr = 71 + lane
        x, y = r[x_addr], 141 - 24 * lane
        if 0 < x < 160:
            if submarine:
                _object(objects, r, f"enemy_missile_l{lane}", "enemy_missile",
                        (x + 3, y + 4, 6, 4), [x_addr, 89 + lane], lane=lane)
            else:
                _object(objects, r, f"diver_l{lane}", "diver", (x, y, 8, 11),
                        [x_addr, 89 + lane], lane=lane)
        for slot in range(3):
            if r[36 + lane] & (1 << (2 - slot)):
                ex = (r[30 + lane] + 16 * slot) % 256
                kind = "enemy_submarine" if submarine else "shark"
                ey = y if submarine else y + r[93] - 4
                _object(objects, r, f"enemy_l{lane}_s{slot}", kind,
                        (ex, ey, 8, 11 if submarine else 7),
                        [30 + lane, 36 + lane, 89 + lane, 93], lane=lane)
    if r[60] >= 2 and r[118] < 160:
        _object(objects, r, "surface_submarine", "enemy_submarine",
                (r[118], 45, 8, 11), [60, 118])
    if 0 < r[103] < 160:
        _object(objects, r, "player_missile", "player_missile",
                (r[103], r[97] + 40, 8, 1), [103, 97])
    _object(objects, r, "oxygen_bar", "oxygen_bar", (49, 170, min(r[102], 63), 5), [102])
    for i in range(min(int(r[62]), 6)):
        _object(objects, r, f"collected_diver_{i}", "collected_diver",
                (58 + 8 * i, 178, 8, 9), [62])
    scalars["visible_divers"] = sum(o["kind"] == "diver" for o in objects)
    return scalars


def _frostbite(r, objects, warnings):
    raw = int(r[77])
    stage = 0 if raw == 255 else raw + 1 if raw < 16 else None
    scalars = dict(igloo_stage=stage, igloo_raw=raw, reserve_lives=int(r[76]),
                   temperature_raw=int(r[101]), temperature=_bcd_or_none(r[101]), sinking_raw=int(r[107]),
                   death_animation_raw=int(r[106]), floe_layout_raw=int(r[30]))
    if stage is None:
        warnings.append(f"Unknown igloo byte {raw}; stage left null")
    if r[106] != 26 and r[107] < 18:
        _object(objects, r, "player", "player",
                (r[102], r[100] + 29 + r[107], 8, 18 - r[107]), [100, 102, 106, 107])
    if r[104] != 140:
        _object(objects, r, "bear", "bear", (r[104] + 3, 58, 14, 16), [104])
    if stage:
        # Width changes in the first row, then subsequent rows grow upward.
        heights = (4, 4, 4, 4, 8, 8, 8, 8, 13, 13, 13, 13, 16, 16, 20, 20)
        height = heights[raw]
        _object(objects, r, "igloo", "igloo",
                (112, 55 - height, min(32, 8 * stage), height), [77], stage=stage)
        if stage == 16:
            _object(objects, r, "igloo_door", "igloo_door", (123, 47, 8, 8), [77])
    for lane in range(4):  # bottom to top, same convention as Seaquest
        t = int(r[35 + lane])
        kind = ("bird" if t in (18, 26) else
                "crab" if 33 < t < 39 or 49 < t < 55 else
                "clam" if 65 < t < 71 or 93 < t < 99 else
                "fish" if 108 < t < 114 or 123 < t < 129 else None)
        if 0 < r[84 + lane] < 160 or r[88 + lane]:
            if kind is None:
                warnings.append(f"Unknown lane {lane} enemy type {t}; omitted")
            else:
                for slot in range(2 if r[88 + lane] else 1):
                    _object(objects, r, f"enemy_l{lane}_s{slot}", kind,
                            (r[84 + lane] + 32 * slot, 160 - 26 * lane,
                             8, 6 if kind == "fish" else 7),
                            [35 + lane, 84 + lane, 88 + lane], lane=lane)
        single = r[30] == 8
        space = abs(int(r[30]) - 8)
        scalars[f"floe_l{lane}_white"] = int(r[43 + lane] == 12)
        for slot in range(3 if single else 6):
            offset = (32 * slot - 8 if single else
                      16 * slot - 8 - max(0, space - 4) if slot % 2 == 0 else
                      16 * slot - 16 + min(space, 4))
            x = (r[31 + lane] + offset) % 160
            _object(objects, r, f"floe_l{lane}_s{slot}", "floe",
                    (x, 174 - 26 * lane, 24 if single else 16, 7),
                    [30, 31 + lane, 43 + lane], lane=lane,
                    white=bool(r[43 + lane] == 12), wraps_right=bool(x + (24 if single else 16) > 160))
    return scalars


def _pong(r, objects, warnings):
    scalars = dict(player_score=int(r[14]), enemy_score=int(r[13]),
                   ball_present=int(r[54] != 0 and r[49] > 49))
    if scalars["ball_present"]:
        _object(objects, r, "ball", "ball", (r[49] - 49, r[54] - 14, 2, 4), [49, 54])
    # Paddle clipping is asymmetric; use the game's separate edge rules.
    for name, x, address, offset, lower in (("player", 140, 51, 13, 192),
                                           ("enemy", 16, 50, 15, 194)):
        if r[address] > 33:
            if r[address] < 34 + offset:
                y, height = 34, r[address] - 33
            elif r[address] > lower:
                y, height = r[address] - offset, lower + 15 - r[address]
            else:
                y, height = r[address] - offset, 15
            _object(objects, r, name, name, (x, y, 4, height), [address])
    return scalars


def _boxing(r, objects, warnings):
    # Displayed scores use BCD: e.g. RAM 0x24 (=36) means 24 points.
    scalars = dict(player_score_raw=int(r[18]), enemy_score_raw=int(r[19]),
                   player_score=_bcd_or_none(r[18]), enemy_score=_bcd_or_none(r[19]),
                   player_left_arm_raw=int(r[55]), player_right_arm_raw=int(r[57]),
                   enemy_left_arm_raw=int(r[61]), enemy_right_arm_raw=int(r[59]))
    if scalars["player_score"] is None or scalars["enemy_score"] is None:
        warnings.append("Non-BCD boxing score (e.g. special end-of-game code); decoded score left null")
    _object(objects, r, "player", "player", (r[32] + 5, r[34] + 38, 14, 46), [32, 34])
    _object(objects, r, "enemy", "enemy", (r[33] + 4, r[35] + 38, 14, 46), [33, 35])
    return scalars


def extract_state(game, ram):
    """Return JSON-safe scalars, visible objects, source bytes and warnings.

    Each call is independent; shuffled retrieval neighbors are supported.
    An omitted object is absent/offscreen according to the RAM rules, not (0, 0).
    Visual sprite flicker is not a reliable RAM absence signal.
    """
    game = canonical_game(game)
    r = validate_ram(ram)
    objects, warnings = [], []
    scalars = {"seaquest": _seaquest, "frostbite": _frostbite,
               "pong": _pong, "boxing": _boxing}[game](r, objects, warnings)
    return dict(game=game, scalars=scalars, objects=objects, warnings=warnings)


def numeric_features(state):
    """Flat values in native units; missing positions must remain masked/NaN.

    This is an export format, NOT a calibrated semantic distance. For set-like
    objects match objects by kind/position, rather than treating slots as identity.
    """
    features = dict(state["scalars"])
    for obj in state["objects"]:
        prefix = obj["id"]
        features[f"{prefix}.present"] = 1
        features[f"{prefix}.kind.{obj['kind']}"] = 1
        for axis, value in zip(("x", "y", "w", "h"), obj["xywh"]):
            features[f"{prefix}.{axis}"] = value
    return features
