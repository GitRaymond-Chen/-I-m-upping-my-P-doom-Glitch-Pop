"""Scene registry: scene key -> function(ctx)."""
from . import part1, part2, part3, part4

SCENE_FNS = {
    "intro": part1.intro,
    "L01": part1.L01, "L02": part1.L02, "L03": part1.L03, "L04": part1.L04,
    "L05": part1.L05, "L06": part1.L06, "L07": part1.L07, "L08": part1.L08,
    "L09": part1.L09,
}
SCENE_FNS.update(part2.SCENES)
SCENE_FNS.update(part3.SCENES)
SCENE_FNS.update(part4.SCENES)
