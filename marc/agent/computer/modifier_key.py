from typing import final

from aenum import MultiValueEnum  # pyright: ignore[reportMissingTypeStubs]
from pynput.keyboard import Key


@final
class ModifierKey(MultiValueEnum):
    alt = "alt"
    backspace = "backspace"
    caps_lock = "caps_lock"
    cmd = "cmd", "command"
    ctrl = "ctrl", "control"
    delete = "delete"
    down = "down"
    end = "end"
    enter = "enter", "return"
    esc = "esc"
    f1 = "f1"
    f2 = "f2"
    f3 = "f3"
    f4 = "f4"
    f5 = "f5"
    f6 = "f6"
    f7 = "f7"
    f8 = "f8"
    f9 = "f9"
    f10 = "f10"
    f11 = "f11"
    f12 = "f12"
    f13 = "f13"
    f14 = "f14"
    f15 = "f15"
    f16 = "f16"
    f17 = "f17"
    f18 = "f18"
    f19 = "f19"
    f20 = "f20"
    home = "home"
    left = "left"
    page_down = "page_down"
    page_up = "page_up"
    right = "right"
    shift = "shift"
    space = "space", " "
    tab = "tab"
    up = "up"
    media_play_pause = "media_play_pause"
    media_volume_mute = "media_volume_mute"
    media_volume_down = "media_volume_down"
    media_volume_up = "media_volume_up"
    media_previous = "media_previous"
    media_next = "media_next"


MODIFIER_KEY_MAP = {
    ModifierKey.alt: Key.alt,
    ModifierKey.backspace: Key.backspace,
    ModifierKey.caps_lock: Key.caps_lock,
    ModifierKey.cmd: Key.cmd,
    ModifierKey.ctrl: Key.ctrl,
    ModifierKey.delete: Key.delete,
    ModifierKey.down: Key.down,
    ModifierKey.end: Key.end,
    ModifierKey.enter: Key.enter,
    ModifierKey.esc: Key.esc,
    ModifierKey.f1: Key.f1,
    ModifierKey.f2: Key.f2,
    ModifierKey.f3: Key.f3,
    ModifierKey.f4: Key.f4,
    ModifierKey.f5: Key.f5,
    ModifierKey.f6: Key.f6,
    ModifierKey.f7: Key.f7,
    ModifierKey.f8: Key.f8,
    ModifierKey.f9: Key.f9,
    ModifierKey.f10: Key.f10,
    ModifierKey.f11: Key.f11,
    ModifierKey.f12: Key.f12,
    ModifierKey.f13: Key.f13,
    ModifierKey.f14: Key.f14,
    ModifierKey.f15: Key.f15,
    ModifierKey.f16: Key.f16,
    ModifierKey.f17: Key.f17,
    ModifierKey.f18: Key.f18,
    ModifierKey.f19: Key.f19,
    ModifierKey.f20: Key.f20,
    ModifierKey.home: Key.home,
    ModifierKey.left: Key.left,
    ModifierKey.page_down: Key.page_down,
    ModifierKey.page_up: Key.page_up,
    ModifierKey.right: Key.right,
    ModifierKey.shift: Key.shift,
    ModifierKey.space: Key.space,
    ModifierKey.tab: Key.tab,
    ModifierKey.up: Key.up,
    ModifierKey.media_play_pause: Key.media_play_pause,
    ModifierKey.media_volume_mute: Key.media_volume_mute,
    ModifierKey.media_volume_down: Key.media_volume_down,
    ModifierKey.media_volume_up: Key.media_volume_up,
    ModifierKey.media_previous: Key.media_previous,
}
