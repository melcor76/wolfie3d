#!/usr/bin/env python3
"""
Vibe Wolf (Python + PyOpenGL) — GL-renderer
-------------------------------------------
Denne varianten beholder logikken (kart, DDA-raycasting, input, sprites),
men tegner ALT med OpenGL (GPU). Vegger og sprites blir teksturerte quads,
og vi bruker depth-test i GPU for korrekt okklusjon (ingen CPU zbuffer).

Avhengigheter:
  - pygame >= 2.1 (for vindu/input)
  - PyOpenGL, PyOpenGL-accelerate
  - numpy

Kjør:
  python wolfie3d_gl.py

Taster:
  - WASD / piltaster: bevegelse
  - Q/E eller ← → : rotasjon
  - SPACE / venstre mus: skyte
  - ESC: avslutt
"""

from __future__ import annotations

import math
import sys
import random
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import pygame
from OpenGL import GL as gl

if TYPE_CHECKING:  # kun for typing hints
    from collections.abc import Sequence

# ---------- Konfig ----------
WIDTH, HEIGHT = 1024, 600
HALF_W, HALF_H = WIDTH // 2, HEIGHT // 2
FPS = 60

# Kamera/FOV
FOV = 66 * math.pi / 180.0
PLANE_LEN = math.tan(FOV / 2)

# Bevegelse
MOVE_SPEED = 3.0  # enheter/sek
ROT_SPEED = 2.0  # rad/sek
STRAFE_SPEED = 2.5

# Tekstur-størrelse brukt på GPU (proseduralt generert)
TEX_W = TEX_H = 256

# Depth mapping (lineær til [0..1] for gl_FragDepth)
FAR_PLANE = 100.0

# Spiller-HP
PLAYER_MAX_HP = 100
PLAYER_CONTACT_DPS = 15.0  # skade per sekund per fiende innenfor radius

# ----------- Lyd utils -----------
def try_init_mixer() -> bool:
    """Initialize pygame.mixer once; return True if ok.
    Safe to call multiple times. Avoid crashing on systems without audio.
    """
    try:
        if not pygame.mixer.get_init():
            # Higher quality output: 44.1 kHz, 16-bit, mono, low-latency buffer
            pygame.mixer.pre_init(44100, -16, 1, 256)
            pygame.mixer.init()
        return True
    except Exception:
        return False


def _get_mixer_sr(default: int = 44100) -> int:
    """Return current mixer sample rate or default."""
    try:
        init = pygame.mixer.get_init()
        if init:
            return int(init[0])
    except Exception:
        pass
    return default


def _find_sounds_dir() -> Path | None:
    """Locate assets/sounds folder from likely paths."""
    try:
        here = Path(__file__).resolve()
    except Exception:
        return None
    candidates = [
        here.parent / "assets" / "sounds",  # src/wolfie3d/assets/sounds
        here.parent.parent / "assets" / "sounds",  # src/assets/sounds
        here.parent.parent.parent / "assets" / "sounds",  # wolfie3d/assets/sounds
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def load_sound_asset(*filenames: str) -> pygame.mixer.Sound | None:
    """Try to load the first available sound (wav/ogg) from assets/sounds."""
    if not try_init_mixer():
        return None
    base = _find_sounds_dir()
    if not base:
        return None
    tried: list[Path] = []
    for fname in filenames:
        # Accept given name and auto-switch between .wav/.ogg
        name = fname
        p = base / name
        tried.append(p)
        if p.exists():
            try:
                return pygame.mixer.Sound(p.as_posix())
            except Exception:
                continue
        # try alt extension swap
        if name.endswith(".wav"):
            alt = base / (name[:-4] + ".ogg")
        elif name.endswith(".ogg"):
            alt = base / (name[:-4] + ".wav")
        else:
            alt = None
        if alt and alt.exists():
            try:
                return pygame.mixer.Sound(alt.as_posix())
            except Exception:
                continue
    return None


def make_pickup_sound() -> pygame.mixer.Sound | None:
    """Load pickup sound asset if present, else generate a short ping."""
    # Prefer asset: pickup.(wav|ogg)
    snd = load_sound_asset("pickup.wav", "pickup.ogg", "item_pickup.wav", "item_pickup.ogg")
    if snd is not None:
        return snd
    if not try_init_mixer():
        return None
    sr = _get_mixer_sr()
    dur = 0.12
    n = int(sr * dur)
    # Exponentially rising blip (simple chirp) with quick decay
    t = np.linspace(0, dur, n, endpoint=False)
    f0, f1 = 600.0, 1200.0
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * (t * t / dur))
    env = np.exp(-t * 20.0)
    wave = 0.6 * np.sin(phase) * env
    # convert to 16-bit mono
    arr = np.int16(np.clip(wave, -1.0, 1.0) * 32767)
    try:
        return pygame.mixer.Sound(buffer=arr.tobytes())
    except Exception:
        return None


def make_shot_sound() -> pygame.mixer.Sound | None:
    """Load shot sound asset if present, else generate a short pew."""
    snd = load_sound_asset("shot.wav", "shot.ogg")
    if snd is not None:
        return snd
    if not try_init_mixer():
        return None
    sr = _get_mixer_sr()
    dur = 0.08
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    # Square-ish wave via sign(sin) with slight pitch drop
    f0, f1 = 1600.0, 900.0
    freq = f0 + (f1 - f0) * (t / dur)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    wave = np.sign(np.sin(phase)) * (0.5)  # harsh
    # envelope: super fast attack, very fast decay
    env = np.minimum(t / 0.004, 1.0) * np.exp(-t * 40.0)
    y = wave * env
    arr = np.int16(np.clip(y, -1.0, 1.0) * 32767)
    try:
        return pygame.mixer.Sound(buffer=arr.tobytes())
    except Exception:
        return None


def make_enemy_death_sound() -> pygame.mixer.Sound | None:
    """Prefer enemy death asset, else procedural cry."""
    snd = load_sound_asset("enemy_death.wav", "enemy_death.ogg")
    if snd is not None:
        return snd
    return _make_death_cry_sound(variant="enemy")


def make_player_death_sound() -> pygame.mixer.Sound | None:
    """Prefer player death asset, else procedural cry."""
    snd = load_sound_asset("player_death.wav", "player_death.ogg")
    if snd is not None:
        return snd
    return _make_death_cry_sound(variant="player")


def _make_death_cry_sound(variant: str = "enemy") -> pygame.mixer.Sound | None:
    """Procedural vocal-ish death cry using pitch-drop saw + formant tones + noise + distortion."""
    if not try_init_mixer():
        return None
    sr = _get_mixer_sr()
    if variant == "player":
        dur = 0.7
        f0_start, f0_end = 240.0, 90.0
        grit = 0.85
    else:
        dur = 0.45
        f0_start, f0_end = 300.0, 120.0
        grit = 0.75
    n = int(sr * dur)
    if n <= 16:
        return None
    t = np.linspace(0, dur, n, endpoint=False)
    # Pitch envelope with a touch of vibrato
    vib = 0.03 * np.sin(2 * np.pi * 6.5 * t)
    freq = (f0_start + (f0_end - f0_start) * (t / dur)) * (1.0 + vib)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    frac = np.mod(phase / (2 * np.pi), 1.0)
    saw = 2.0 * frac - 1.0
    # Vowel-like overtones ("AH") as additional sin components (formant hints)
    f1, f2, f3 = 800.0, 1150.0, 2900.0
    vowel = 0.25 * np.sin(2 * np.pi * f1 * t) + 0.18 * np.sin(2 * np.pi * f2 * t) + 0.12 * np.sin(2 * np.pi * f3 * t)
    # Breath/noise component
    rng = np.random.default_rng(20250829)
    noise = rng.standard_normal(n) * 0.1 * np.exp(-t * 10.0)
    # Combine and apply soft clipping for grit
    raw = 0.7 * saw + vowel + noise
    y = np.tanh(grit * raw)
    # Amplitude envelope: fast attack, curved decay
    attack = np.minimum(t / 0.02, 1.0)
    decay = np.exp(-t * (3.0 if variant == "player" else 5.0))
    env = attack * decay
    out = y * env * 0.9
    # Normalize and convert
    out /= (np.max(np.abs(out)) + 1e-9)
    arr = np.int16(np.clip(out, -1.0, 1.0) * 32767)
    try:
        return pygame.mixer.Sound(buffer=arr.tobytes())
    except Exception:
        return None
PLAYER_CONTACT_RADIUS = 0.6

# Kart (0=tomt, >0=veggtype/tekstur-id)
MAP: list[list[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 1],
    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 1],
    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 1],
    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 1],
    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 1],
    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 1],
    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 0, 0, 4, 4, 4, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
MAP_W = len(MAP[0])
MAP_H = len(MAP)

# Startpos og retning
player_x = 3.5
player_y = 10.5
dir_x, dir_y = 1.0, 0.0
plane_x, plane_y = 0.0, PLANE_LEN


# ---------- Hjelpere ----------
def in_map(ix: int, iy: int) -> bool:
    return 0 <= ix < MAP_W and 0 <= iy < MAP_H


def is_wall(ix: int, iy: int) -> bool:
    return in_map(ix, iy) and MAP[iy][ix] > 0


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ---------- Prosjektil ----------
class Bullet:
    def __init__(self, x: float, y: float, vx: float, vy: float) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.alive = True
        self.age = 0.0
        self.height_param = 0.2  # 0..~0.65 (stiger visuelt)

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt
        if is_wall(int(nx), int(ny)):
            self.alive = False
            return
        self.x, self.y = nx, ny
        self.age += dt
        self.height_param = min(0.65, self.height_param + 0.35 * dt)


class AmmoBox:
    def __init__(self, x: float, y: float, amount: int) -> None:
        self.x = x
        self.y = y
        self.amount = amount
        self.alive = True
        self.radius = 0.3
        self.height_param = 0.35


class PickupPop:
    """Transient visual effect when picking up ammo: quick scale 'pop'."""
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.age = 0.0
        self.dur = 0.20  # seconds
        self.alive = True

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        self.age += dt
        if self.age >= self.dur:
            self.alive = False

# ---------- Fiende ----------
# Enemy types
ENEMY_BLACK_HELMET = 0  # 3 HP - black uniform with helmet
ENEMY_GREY_HELMET = 1  # 2 HP - grey uniform with helmet
ENEMY_GREY_NO_HELMET = 2  # 1 HP - grey uniform without helmet

# Poeng per fiendetype
ENEMY_SCORE: dict[int, int] = {
    ENEMY_GREY_NO_HELMET: 100,  # lett
    ENEMY_GREY_HELMET: 200,  # middels
    ENEMY_BLACK_HELMET: 300,  # vanskelig
}


class Enemy:
    def __init__(self, x: float, y: float, enemy_type: int = ENEMY_GREY_NO_HELMET) -> None:
        self.x = x
        self.y = y
        self.enemy_type = enemy_type
        self.alive = True
        self.radius = 0.35  # kollisjon/hitbox i kart-enheter
        self.speed = 1.6  # enheter/sek (enkel jakt)
        self.height_param = 0.5  # hvor høyt sprite sentreres i skjerm

        # Set HP/texture by type
        if self.enemy_type == ENEMY_BLACK_HELMET:
            self.max_hp = 3
            self.texture_id = 200
        elif self.enemy_type == ENEMY_GREY_HELMET:
            self.max_hp = 2
            self.texture_id = 201
        else:  # ENEMY_GREY_NO_HELMET
            self.max_hp = 1
            self.texture_id = 202

        self.hp = self.max_hp
        # kort visuell feedback når fienden blir truffet
        self.hurt_t: float = 0.0  # sekunder igjen med "rød blink"

    def take_damage(self, damage: int = 1) -> bool:
        """Take damage and return True if enemy died"""
        if not self.alive:
            return False
        self.hp -= damage
        self.hurt_t = 0.18  # ~3 frames at 60 FPS
        if self.hp <= 0:
            self.alive = False
            return True
        return False

    def _try_move(self, nx: float, ny: float) -> None:
        # enkel vegg-kollisjon (sirkulær hitbox mot grid)
        # prøv X:
        if not is_wall(int(nx), int(self.y)):
            self.x = nx
        # prøv Y:
        if not is_wall(int(self.x), int(ny)):
            self.y = ny

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        # fade ut "hit flash"
        if self.hurt_t > 0.0:
            self.hurt_t = max(0.0, self.hurt_t - dt)
        # enkel "chase": gå mot spilleren, stopp om rett foran vegg
        dx = player_x - self.x
        dy = player_y - self.y
        dist = math.hypot(dx, dy) + 1e-9
        # la fiender komme nær nok til at kontakt-skade starter
        # stopp litt INNENFOR skade-radius
        stop_dist = max(0.35, PLAYER_CONTACT_RADIUS - 0.05)
        if dist > stop_dist:
            ux, uy = dx / dist, dy / dist
            step = self.speed * dt
            self._try_move(self.x + ux * step, self.y + uy * step)


# ---------- Prosedural tekstur (pygame.Surface) ----------
def make_brick_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H))
    surf.fill((150, 40, 40))
    mortar = (200, 200, 200)
    brick_h = TEX_H // 4
    brick_w = TEX_W // 4
    for row in range(0, TEX_H, brick_h):
        offset = 0 if (row // brick_h) % 2 == 0 else brick_w // 2
        for col in range(0, TEX_W, brick_w):
            rect = pygame.Rect((col + offset) % TEX_W, row, brick_w - 1, brick_h - 1)
            pygame.draw.rect(surf, (165, 52, 52), rect)
    for y in range(0, TEX_H, brick_h):
        pygame.draw.line(surf, mortar, (0, y), (TEX_W, y))
    for x in range(0, TEX_W, brick_w):
        pygame.draw.line(surf, mortar, (x, 0), (x, TEX_H))
    return surf


def make_stone_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H))
    base = (110, 110, 120)
    surf.fill(base)
    for y in range(TEX_H):
        for x in range(TEX_W):
            if ((x * 13 + y * 7) ^ (x * 3 - y * 5)) & 15 == 0:
                c = 90 + ((x * y) % 40)
                surf.set_at((x, y), (c, c, c))
    for i in range(5):
        pygame.draw.line(surf, (80, 80, 85), (i * 12, 0), (TEX_W - 1, TEX_H - 1 - i * 6), 1)
    return surf


def make_wood_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H))
    for y in range(TEX_H):
        for x in range(TEX_W):
            v = int(120 + 40 * math.sin((x + y * 0.5) * 0.12) + 20 * math.sin(y * 0.3))
            v = max(60, min(200, v))
            surf.set_at((x, y), (140, v, 60))
    for x in range(0, TEX_W, TEX_W // 4):
        pygame.draw.line(surf, (90, 60, 30), (x, 0), (x, TEX_H))
    return surf


def make_metal_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H), pygame.SRCALPHA)
    base = (140, 145, 150, 255)
    surf.fill(base)
    for y in range(8, TEX_H, 16):
        for x in range(8, TEX_W, 16):
            pygame.draw.circle(surf, (90, 95, 100, 255), (x, y), 2)
    for y in range(TEX_H):
        shade = 130 + (y % 8) * 2
        pygame.draw.line(surf, (shade, shade, shade + 5, 255), (0, y), (TEX_W, y), 1)
    return surf


def make_bullet_texture() -> pygame.Surface:
    surf = pygame.Surface((32, 32), pygame.SRCALPHA)
    pygame.draw.circle(surf, (255, 240, 150, 220), (16, 16), 8)
    pygame.draw.circle(surf, (255, 255, 255, 255), (13, 13), 3)
    return surf


def make_ammo_box_texture() -> pygame.Surface:
    s = pygame.Surface((64, 64), pygame.SRCALPHA)
    # box
    pygame.draw.rect(s, (70, 120, 70, 255), (6, 10, 52, 44), border_radius=6)
    # plus sign
    pygame.draw.rect(s, (230, 240, 230, 255), (30, 18, 4, 28))
    pygame.draw.rect(s, (230, 240, 230, 255), (22, 30, 20, 4))
    # subtle border
    pygame.draw.rect(s, (40, 70, 40, 255), (6, 10, 52, 44), width=2, border_radius=6)
    return s


# ---------- OpenGL utils ----------
VS_SRC = """
#version 330 core
layout (location = 0) in vec2 in_pos;    // NDC -1..1
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec3 in_col;    // per-vertex farge (for dimming/overlay)
layout (location = 3) in float in_depth; // 0..1 depth (0 nær, 1 fjern)

out vec2 v_uv;
out vec3 v_col;
out float v_depth;

void main() {
    v_uv = in_uv;
    v_col = in_col;
    v_depth = in_depth;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FS_SRC = """
#version 330 core
in vec2 v_uv;
in vec3 v_col;
in float v_depth;

out vec4 fragColor;

uniform sampler2D uTexture;
uniform bool uUseTexture;

void main() {
    vec4 base = vec4(1.0);
    if (uUseTexture) {
        base = texture(uTexture, v_uv);
        if (base.a < 0.01) discard; // alpha for sprites
    }
    vec3 rgb = base.rgb * v_col;
    fragColor = vec4(rgb, base.a);
    // Skriv eksplisitt dybde (lineær i [0..1])
    gl_FragDepth = clamp(v_depth, 0.0, 1.0);
}
"""


def compile_shader(src: str, stage: int) -> int:
    sid = gl.glCreateShader(stage)
    gl.glShaderSource(sid, src)
    gl.glCompileShader(sid)
    status = gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS)
    if status != gl.GL_TRUE:
        log = gl.glGetShaderInfoLog(sid).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return sid


def make_program(vs_src: str, fs_src: str) -> int:
    vs = compile_shader(vs_src, gl.GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    ok = gl.glGetProgramiv(prog, gl.GL_LINK_STATUS)
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    if ok != gl.GL_TRUE:
        log = gl.glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    return prog


def surface_to_texture(surf: pygame.Surface) -> int:
    """Laster pygame.Surface til GL_TEXTURE_2D (RGBA8). Returnerer texture id."""
    data = pygame.image.tostring(surf.convert_alpha(), "RGBA", True)
    w, h = surf.get_width(), surf.get_height()
    tid = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data
    )
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tid


def make_white_texture() -> int:
    surf = pygame.Surface((1, 1), pygame.SRCALPHA)
    surf.fill((255, 255, 255, 255))
    return surface_to_texture(surf)


def make_enemy_texture() -> pygame.Surface:
    s = pygame.Surface((256, 256), pygame.SRCALPHA)
    # kropp
    pygame.draw.rect(s, (140, 140, 145, 255), (100, 80, 56, 120), border_radius=6)
    # hode - no helmet (grey uniform without helmet)
    pygame.draw.circle(s, (220, 200, 180, 255), (128, 70), 26)
    # “arm”
    pygame.draw.rect(s, (140, 140, 145, 255), (86, 110, 24, 16))
    pygame.draw.rect(s, (140, 140, 145, 255), (146, 110, 24, 16))
    return s


def make_enemy_black_helmet_texture() -> pygame.Surface:
    """Black uniform with helmet - 3 HP"""
    s = pygame.Surface((256, 256), pygame.SRCALPHA)
    # kropp - black uniform
    pygame.draw.rect(s, (40, 40, 50, 255), (100, 80, 56, 120), border_radius=6)
    # hode
    pygame.draw.circle(s, (220, 200, 180, 255), (128, 70), 26)
    # helmet - more prominent
    pygame.draw.arc(s, (20, 20, 30, 255), (92, 40, 72, 50), 3.14, 0, 8)
    pygame.draw.ellipse(s, (20, 20, 30, 255), (100, 44, 56, 32))
    # "arm" - black
    pygame.draw.rect(s, (40, 40, 50, 255), (86, 110, 24, 16))
    pygame.draw.rect(s, (40, 40, 50, 255), (146, 110, 24, 16))
    return s


def make_enemy_grey_helmet_texture() -> pygame.Surface:
    """Grey uniform with helmet - 2 HP"""
    s = pygame.Surface((256, 256), pygame.SRCALPHA)
    # kropp - darker grey uniform
    pygame.draw.rect(s, (100, 100, 105, 255), (100, 80, 56, 120), border_radius=6)
    # hode
    pygame.draw.circle(s, (220, 200, 180, 255), (128, 70), 26)
    # helmet - grey
    pygame.draw.arc(s, (80, 80, 85, 255), (92, 40, 72, 45), 3.14, 0, 6)
    pygame.draw.ellipse(s, (80, 80, 85, 255), (102, 46, 52, 28))
    # "arm" - darker grey
    pygame.draw.rect(s, (100, 100, 105, 255), (86, 110, 24, 16))
    pygame.draw.rect(s, (100, 100, 105, 255), (146, 110, 24, 16))
    return s


# ---------- GL Renderer state ----------
from pathlib import Path
import os
import pygame
from OpenGL import GL as gl


# ---------- GL Renderer state ----------
class GLRenderer:
    def __init__(self) -> None:
        # Shader program
        self.prog = make_program(VS_SRC, FS_SRC)
        gl.glUseProgram(self.prog)
        self.uni_tex = gl.glGetUniformLocation(self.prog, "uTexture")
        self.uni_use_tex = gl.glGetUniformLocation(self.prog, "uUseTexture")
        gl.glUniform1i(self.uni_tex, 0)

        # VAO/VBO (dynamisk buffer per draw)
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        stride = 8 * 4  # 8 float32 per vertex
        # in_pos (loc 0): 2 floats
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0))
        # in_uv (loc 1): 2 floats
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(2 * 4))
        # in_col (loc 2): 3 floats
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(4 * 4))
        # in_depth (loc 3): 1 float
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(3, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(7 * 4))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # Teksturer
        self.white_tex = make_white_texture()
        self.textures: dict[int, int] = {}  # tex_id -> GL texture

        # Last fra assets hvis tilgjengelig, ellers fall tilbake til proseduralt
        self.load_textures()

        # GL state
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

    # ---------- teksturhjelpere ----------
    @staticmethod
    def _scale_if_needed(surf: pygame.Surface, size: int = 512) -> pygame.Surface:
        if surf.get_width() != size or surf.get_height() != size:
            surf = pygame.transform.smoothscale(surf, (size, size))
        return surf

    def _load_texture_file(self, path: str, size: int = 512) -> int:
        surf = pygame.image.load(path).convert_alpha()
        surf = self._scale_if_needed(surf, size)
        return surface_to_texture(surf)

    # ---------- offentlig laster ----------

    def _resolve_textures_base(self) -> Path:
        """
        Finn korrekt assets/textures-katalog robust, uavhengig av hvor vi kjører fra.
        Prøver i rekkefølge:
          - <her>/assets/textures
          - <her>/../assets/textures
          - <her>/../../assets/textures      <-- typisk når koden ligger i src/wolfie3d
          - <cwd>/assets/textures
        """
        here = Path(__file__).resolve().parent
        candidates = [
            here / "assets" / "textures",
            here.parent / "assets" / "textures",
            here.parent.parent / "assets" / "textures",
            Path.cwd() / "assets" / "textures",
        ]
        print("\n[GLRenderer] Prøver å finne assets/textures på disse stedene:")
        for c in candidates:
            print("  -", c)
            if c.exists():
                print("[GLRenderer] FANT:", c)
                return c

        raise FileNotFoundError(
            "Fant ikke assets/textures i noen av kandidatkatalogene over. "
            "Opprett assets/textures på prosjektnivå (samme nivå som src) eller justér stien."
        )

    def load_textures(self) -> None:
        """
        Debug-variant som bruker korrekt prosjekt-rot og feiler høyt hvis filer mangler.
        Forventer: bricks.png, stone.png, wood.png, metal.png i assets/textures/.
        """
        base = self._resolve_textures_base()
        print(f"[GLRenderer] pygame extended image support: {pygame.image.get_extended()}")
        print(f"[GLRenderer] Innhold i {base}: {[p.name for p in base.glob('*')]}")

        files = {
            1: base / "bricks.png",
            2: base / "stone.png",
            3: base / "wood.png",
            4: base / "metal.png",
        }
        missing = [p for p in files.values() if not p.exists()]
        if missing:
            print("[GLRenderer] MANGEL: følgende filer finnes ikke:")
            for m in missing:
                print("  -", m)
            raise FileNotFoundError(
                "Manglende teksturer. Sørg for at filene ligger i assets/textures/"
            )

        def _load(path: Path, size: int = 512) -> int:
            print(f"[GLRenderer] Laster: {path}")
            surf = pygame.image.load(str(path)).convert_alpha()
            if surf.get_width() != size or surf.get_height() != size:
                print(
                    f"[GLRenderer]  - rescale {surf.get_width()}x{surf.get_height()} -> {size}x{size}"
                )
                surf = pygame.transform.smoothscale(surf, (size, size))
            tex_id = surface_to_texture(surf)
            print(f"[GLRenderer]  - OK (GL tex id {tex_id})")
            return tex_id

        self.textures[1] = _load(files[1], 512)
        self.textures[2] = _load(files[2], 512)
        self.textures[3] = _load(files[3], 512)
        self.textures[4] = _load(files[4], 512)

        # Sprite textures
        self.textures[99] = surface_to_texture(make_bullet_texture())
        self.textures[150] = surface_to_texture(make_ammo_box_texture())

        # Enemy sprites (ID 200, 201, 202): prøv fil, ellers prosedural placeholder
        try:
            sprites_dir = self._resolve_textures_base().parent / "sprites"
            enemy_path = sprites_dir / "enemy.png"
            print(f"[GLRenderer] Leter etter enemy sprite i: {enemy_path}")
            if enemy_path.exists():
                self.textures[200] = self._load_texture_file(enemy_path, 512)
                print(f"[GLRenderer] Enemy OK (GL tex id {self.textures[200]})")
                # Use same texture for all types if file exists
                self.textures[201] = self.textures[200]
                self.textures[202] = self.textures[200]
            else:
                # fallback – prosedural fiende textures
                self.textures[200] = surface_to_texture(make_enemy_black_helmet_texture())
                self.textures[201] = surface_to_texture(make_enemy_grey_helmet_texture())
                self.textures[202] = surface_to_texture(make_enemy_texture())
                print("[GLRenderer] Enemy: bruker prosedural sprites")
        except Exception as ex:
            print(f"[GLRenderer] Enemy: FEIL ved lasting ({ex}), bruker prosedural")
            self.textures[200] = surface_to_texture(make_enemy_black_helmet_texture())
            self.textures[201] = surface_to_texture(make_enemy_grey_helmet_texture())
            self.textures[202] = surface_to_texture(make_enemy_texture())

        print("[GLRenderer] Teksturer lastet.\n")

    # ---------- draw ----------
    def draw_arrays(self, verts: np.ndarray, texture: int, use_tex: bool) -> None:
        if verts.size == 0:
            return
        # Set shader program and texture
        gl.glUseProgram(self.prog)
        gl.glUniform1i(self.uni_use_tex, 1 if use_tex else 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture if use_tex else self.white_tex)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_DYNAMIC_DRAW)
        count = verts.shape[0]
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, count)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)


# ---------- Raycasting + bygg GL-verts ----------
def column_ndc(x: int) -> tuple[float, float]:
    """Returnerer venstre/høyre NDC-X for en 1-px bred skjermkolonne."""
    x_left = (2.0 * x) / WIDTH - 1.0
    x_right = (2.0 * (x + 1)) / WIDTH - 1.0
    return x_left, x_right


def y_ndc(y_pix: int) -> float:
    """Konverter skjerm-Y (0 top) til NDC-Y (1 top, -1 bunn)."""
    return 1.0 - 2.0 * (y_pix / float(HEIGHT))


def dim_for_side(side: int) -> float:
    # dim litt på sidevegger (liknende BLEND_MULT tidligere)
    return 0.78 if side == 1 else 1.0


def cast_and_build_wall_batches() -> dict[int, list[float]]:
    batches: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
    for x in range(WIDTH):
        # Raydir
        camera_x = 2.0 * x / WIDTH - 1.0
        ray_dir_x = dir_x + plane_x * camera_x
        ray_dir_y = dir_y + plane_y * camera_x
        map_x = int(player_x)
        map_y = int(player_y)

        delta_dist_x = abs(1.0 / ray_dir_x) if ray_dir_x != 0 else 1e30
        delta_dist_y = abs(1.0 / ray_dir_y) if ray_dir_y != 0 else 1e30

        if ray_dir_x < 0:
            step_x = -1
            side_dist_x = (player_x - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - player_x) * delta_dist_x
        if ray_dir_y < 0:
            step_y = -1
            side_dist_y = (player_y - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - player_y) * delta_dist_y

        hit = 0
        side = 0
        tex_id = 1
        while hit == 0:
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1
            if not in_map(map_x, map_y):
                hit = 1
                tex_id = 1
                break
            if MAP[map_y][map_x] > 0:
                hit = 1
                tex_id = MAP[map_y][map_x]

        if side == 0:
            perp_wall_dist = (map_x - player_x + (1 - step_x) / 2.0) / (
                ray_dir_x if ray_dir_x != 0 else 1e-9
            )
            wall_x = player_y + perp_wall_dist * ray_dir_y
        else:
            perp_wall_dist = (map_y - player_y + (1 - step_y) / 2.0) / (
                ray_dir_y if ray_dir_y != 0 else 1e-9
            )
            wall_x = player_x + perp_wall_dist * ray_dir_x

        wall_x -= math.floor(wall_x)
        # u-koordinat (kontinuerlig) + flip for samsvar med klassisk raycaster
        u = wall_x
        if (side == 0 and ray_dir_x > 0) or (side == 1 and ray_dir_y < 0):
            u = 1.0 - u

        # skjermhøyde på vegg
        line_height = int(HEIGHT / (perp_wall_dist + 1e-6))
        draw_start = max(0, -line_height // 2 + HALF_H)
        draw_end = min(HEIGHT - 1, line_height // 2 + HALF_H)

        # NDC koordinater for 1-px bred stripe
        x_left, x_right = column_ndc(x)
        top_ndc = y_ndc(draw_start)
        bot_ndc = y_ndc(draw_end)

        # Farge-dim (samme på hele kolonnen)
        c = dim_for_side(side)
        r = g = b = c

        # Depth som lineær [0..1] (0 = nærmest)
        depth = clamp01(perp_wall_dist / FAR_PLANE)

        # 2 triangler (6 vertikser). Vertex-layout:
        # [x, y, u, v, r, g, b, depth]
        v = [
            # tri 1
            x_left,
            top_ndc,
            u,
            0.0,
            r,
            g,
            b,
            depth,
            x_left,
            bot_ndc,
            u,
            1.0,
            r,
            g,
            b,
            depth,
            x_right,
            top_ndc,
            u,
            0.0,
            r,
            g,
            b,
            depth,
            # tri 2
            x_right,
            top_ndc,
            u,
            0.0,
            r,
            g,
            b,
            depth,
            x_left,
            bot_ndc,
            u,
            1.0,
            r,
            g,
            b,
            depth,
            x_right,
            bot_ndc,
            u,
            1.0,
            r,
            g,
            b,
            depth,
        ]
        batches.setdefault(tex_id, []).extend(v)
    return batches


def build_fullscreen_background() -> np.ndarray:
    """To store quads (himmel/gulv), farget med vertex-color, tegnes uten tekstur."""
    # Himmel (øverst halvdel)
    sky_col = (40 / 255.0, 60 / 255.0, 90 / 255.0)
    floor_col = (35 / 255.0, 35 / 255.0, 35 / 255.0)
    verts: list[float] = []

    # Quad helper
    def add_quad(x0, y0, x1, y1, col):
        r, g, b = col
        depth = 1.0  # lengst bak
        # u,v er 0 (vi bruker hvit 1x1 tekstur)
        verts.extend(
            [
                x0,
                y0,
                0.0,
                0.0,
                r,
                g,
                b,
                depth,
                x0,
                y1,
                0.0,
                1.0,
                r,
                g,
                b,
                depth,
                x1,
                y0,
                1.0,
                0.0,
                r,
                g,
                b,
                depth,
                x1,
                y0,
                1.0,
                0.0,
                r,
                g,
                b,
                depth,
                x0,
                y1,
                0.0,
                1.0,
                r,
                g,
                b,
                depth,
                x1,
                y1,
                1.0,
                1.0,
                r,
                g,
                b,
                depth,
            ]
        )

    # Koordinater i NDC
    add_quad(-1.0, 1.0, 1.0, 0.0, sky_col)  # øvre halvdel
    add_quad(-1.0, 0.0, 1.0, -1.0, floor_col)  # nedre halvdel
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_sprites_batch(bullets: list[Bullet]) -> np.ndarray:
    """Bygger ett quad per kule i skjermen (billboard), med depth."""
    verts: list[float] = []

    for b in bullets:
        # Transform til kamera-rom
        spr_x = b.x - player_x
        spr_y = b.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue  # bak kamera

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))

        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h  # kvadratisk

        # vertikal offset: "stiger"
        v_shift = int((0.5 - b.height_param) * sprite_h)
        draw_start_y = max(0, -sprite_h // 2 + HALF_H + v_shift)
        draw_end_y = min(HEIGHT - 1, draw_start_y + sprite_h)
        # horisontal
        draw_start_x = -sprite_w // 2 + sprite_screen_x
        draw_end_x = draw_start_x + sprite_w

        # Klipp utenfor skjerm
        if draw_end_x < 0 or draw_start_x >= WIDTH:
            continue
        draw_start_x = max(0, draw_start_x)
        draw_end_x = min(WIDTH - 1, draw_end_x)

        # Konverter til NDC
        x0 = (2.0 * draw_start_x) / WIDTH - 1.0
        x1 = (2.0 * (draw_end_x + 1)) / WIDTH - 1.0
        y0 = y_ndc(draw_start_y)
        y1 = y_ndc(draw_end_y)

        # Depth (basert på trans_y)
        depth = clamp01(trans_y / FAR_PLANE)

        r = g = bcol = 1.0  # ingen ekstra farge-dim
        # u,v: full tekstur
        u0, v0 = 0.0, 0.0
        u1, v1 = 1.0, 1.0

        verts.extend(
            [
                x0,
                y0,
                u0,
                v0,
                r,
                g,
                bcol,
                depth,
                x0,
                y1,
                u0,
                v1,
                r,
                g,
                bcol,
                depth,
                x1,
                y0,
                u1,
                v0,
                r,
                g,
                bcol,
                depth,
                x1,
                y0,
                u1,
                v0,
                r,
                g,
                bcol,
                depth,
                x0,
                y1,
                u0,
                v1,
                r,
                g,
                bcol,
                depth,
                x1,
                y1,
                u1,
                v1,
                r,
                g,
                bcol,
                depth,
            ]
        )

    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_pickups_batch(pickups: list["AmmoBox"]) -> np.ndarray:
    verts: list[float] = []
    for p in pickups:
        if not p.alive:
            continue
        spr_x = p.x - player_x
        spr_y = p.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue
        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))
        # Render pickups smaller (30% of default sprite size)
        sprite_h = max(1, int(abs(HEIGHT / trans_y) * 0.3))
        sprite_w = sprite_h
        v_shift = int((0.5 - p.height_param) * sprite_h)
        draw_start_y = max(0, -sprite_h // 2 + HALF_H + v_shift)
        draw_end_y = min(HEIGHT - 1, draw_start_y + sprite_h)
        draw_start_x = -sprite_w // 2 + sprite_screen_x
        draw_end_x = draw_start_x + sprite_w
        if draw_end_x < 0 or draw_start_x >= WIDTH:
            continue
        draw_start_x = max(0, draw_start_x)
        draw_end_x = min(WIDTH - 1, draw_end_x)

        x0 = (2.0 * draw_start_x) / WIDTH - 1.0
        x1 = (2.0 * (draw_end_x + 1)) / WIDTH - 1.0
        y0 = y_ndc(draw_start_y)
        y1 = y_ndc(draw_end_y)
        depth = clamp01(trans_y / FAR_PLANE)
        r = g = b = 1.0
        u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0
        verts.extend(
            [
                x0,
                y0,
                u0,
                v0,
                r,
                g,
                b,
                depth,
                x0,
                y1,
                u0,
                v1,
                r,
                g,
                b,
                depth,
                x1,
                y0,
                u1,
                v0,
                r,
                g,
                b,
                depth,
                x1,
                y0,
                u1,
                v0,
                r,
                g,
                b,
                depth,
                x0,
                y1,
                u0,
                v1,
                r,
                g,
                b,
                depth,
                x1,
                y1,
                u1,
                v1,
                r,
                g,
                b,
                depth,
            ]
        )
    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_pickup_pops_batch(pops: list["PickupPop"]) -> np.ndarray:
    """Billboard tiny pop quads with quick scale-up and fade-out."""
    verts: list[float] = []
    for fx in pops:
        if not fx.alive:
            continue
        spr_x = fx.x - player_x
        spr_y = fx.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue
        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))
        base = max(1, int(abs(HEIGHT / trans_y) * 0.2))  # very small base size
        # Scale curve: overshoot then settle quickly (0..dur)
        t = max(0.0, min(1.0, fx.age / fx.dur))
        scale = 0.6 + 0.8 * (1.0 - (1.0 - t) * (1.0 - t))  # ease-out to ~1.4x
        sprite_h = max(1, int(base * scale))
        sprite_w = sprite_h
        v_shift = int((0.5 - 0.45) * sprite_h)
        draw_start_y = max(0, -sprite_h // 2 + HALF_H + v_shift)
        draw_end_y = min(HEIGHT - 1, draw_start_y + sprite_h)
        draw_start_x = -sprite_w // 2 + sprite_screen_x
        draw_end_x = draw_start_x + sprite_w
        if draw_end_x < 0 or draw_start_x >= WIDTH:
            continue
        draw_start_x = max(0, draw_start_x)
        draw_end_x = min(WIDTH - 1, draw_end_x)

        x0 = (2.0 * draw_start_x) / WIDTH - 1.0
        x1 = (2.0 * (draw_end_x + 1)) / WIDTH - 1.0
        y0 = y_ndc(draw_start_y)
        y1 = y_ndc(draw_end_y)
        depth = clamp01(trans_y / FAR_PLANE)
        # subtle fade-out
        a = 1.0 - t
        r = g = b = 0.9 * a + 0.1
        u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0
        verts.extend(
            [
                x0, y0, u0, v0, r, g, b, depth,
                x0, y1, u0, v1, r, g, b, depth,
                x1, y0, u1, v0, r, g, b, depth,
                x1, y0, u1, v0, r, g, b, depth,
                x0, y1, u0, v1, r, g, b, depth,
                x1, y1, u1, v1, r, g, b, depth,
            ]
        )
    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_enemies_batch_by_type(enemies: list["Enemy"], enemy_type: int) -> np.ndarray:
    """Build batch for enemies of a specific type"""
    verts: list[float] = []
    for e in enemies:
        if not e.alive or e.enemy_type != enemy_type:
            continue
        spr_x = e.x - player_x
        spr_y = e.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))
        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h  # kvadratisk
        v_shift = int((0.5 - e.height_param) * sprite_h)

        draw_start_y = max(0, -sprite_h // 2 + HALF_H + v_shift)
        draw_end_y = min(HEIGHT - 1, draw_start_y + sprite_h)
        draw_start_x = -sprite_w // 2 + sprite_screen_x
        draw_end_x = draw_start_x + sprite_w
        if draw_end_x < 0 or draw_start_x >= WIDTH:
            continue

        draw_start_x = max(0, draw_start_x)
        draw_end_x = min(WIDTH - 1, draw_end_x)

        x0 = (2.0 * draw_start_x) / WIDTH - 1.0
        x1 = (2.0 * (draw_end_x + 1)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (draw_start_y / HEIGHT)
        y1 = 1.0 - 2.0 * (draw_end_y / HEIGHT)

        depth = clamp01(trans_y / FAR_PLANE)
        # base-farge per type (multipliseres med tekstur)
        if e.enemy_type == ENEMY_BLACK_HELMET:  # 3 HP
            base_r, base_g, base_b = 0.15, 0.15, 0.18  # nær svart
        elif e.enemy_type == ENEMY_GREY_HELMET:  # 2 HP
            base_r, base_g, base_b = 0.55, 0.55, 0.60  # tydelig grå
        else:  # ENEMY_GREY_NO_HELMET, 1 HP
            base_r, base_g, base_b = 0.95, 0.80, 0.80  # lys/varm

        # bland inn "hit flash" (mot rød) når skadet
        if e.hurt_t > 0.0:
            t = min(1.0, e.hurt_t / 0.18)
            hit_r, hit_g, hit_b = 1.0, 0.3, 0.3
            r = base_r + (hit_r - base_r) * t
            g = base_g + (hit_g - base_g) * t
            b = base_b + (hit_b - base_b) * t
        else:
            r, g, b = base_r, base_g, base_b

        ENEMY_V_FLIP = True  # sett False hvis den blir riktig uten flip
        if ENEMY_V_FLIP:
            u0, v0, u1, v1 = 0.0, 1.0, 1.0, 0.0
        else:
            u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0

        verts.extend(
            [
                x0,
                y0,
                u0,
                v0,
                r,
                g,
                b,
                depth,
                x0,
                y1,
                u0,
                v1,
                r,
                g,
                b,
                depth,
                x1,
                y0,
                u1,
                v0,
                r,
                g,
                b,
                depth,
                x1,
                y0,
                u1,
                v0,
                r,
                g,
                b,
                depth,
                x0,
                y1,
                u0,
                v1,
                r,
                g,
                b,
                depth,
                x1,
                y1,
                u1,
                v1,
                r,
                g,
                b,
                depth,
            ]
        )

    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_enemies_batch(enemies: list["Enemy"]) -> np.ndarray:
    """Legacy function - builds all enemies together"""
    verts: list[float] = []
    for e in enemies:
        if not e.alive:
            continue
        spr_x = e.x - player_x
        spr_y = e.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))
        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h  # kvadratisk
        v_shift = int((0.5 - e.height_param) * sprite_h)

        draw_start_y = max(0, -sprite_h // 2 + HALF_H + v_shift)
        draw_end_y = min(HEIGHT - 1, draw_start_y + sprite_h)
        draw_start_x = -sprite_w // 2 + sprite_screen_x
        draw_end_x = draw_start_x + sprite_w
        if draw_end_x < 0 or draw_start_x >= WIDTH:
            continue

        draw_start_x = max(0, draw_start_x)
        draw_end_x = min(WIDTH - 1, draw_end_x)

        x0 = (2.0 * draw_start_x) / WIDTH - 1.0
        x1 = (2.0 * (draw_end_x + 1)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (draw_start_y / HEIGHT)
        y1 = 1.0 - 2.0 * (draw_end_y / HEIGHT)

        depth = clamp01(trans_y / FAR_PLANE)
        r = g = b = 1.0

        ENEMY_V_FLIP = True  # sett False hvis den blir riktig uten flip
        if ENEMY_V_FLIP:
            u0, v0, u1, v1 = 0.0, 1.0, 1.0, 0.0
        else:
            u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0

        verts.extend(
            [
                x0,
                y0,
                u0,
                v0,
                r,
                g,
                b,
                depth,
                x0,
                y1,
                u0,
                v1,
                r,
                g,
                b,
                depth,
                x1,
                y0,
                u1,
                v0,
                r,
                g,
                b,
                depth,
                x1,
                y0,
                u1,
                v0,
                r,
                g,
                b,
                depth,
                x0,
                y1,
                u0,
                v1,
                r,
                g,
                b,
                depth,
                x1,
                y1,
                u1,
                v1,
                r,
                g,
                b,
                depth,
            ]
        )

    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_crosshair_quads(size_px: int = 8, thickness_px: int = 2) -> np.ndarray:
    """To små rektangler (horisontalt/vertikalt), sentrert i skjermen."""
    verts: list[float] = []

    def rect_ndc(cx, cy, w, h):
        x0 = (2.0 * (cx - w)) / WIDTH - 1.0
        x1 = (2.0 * (cx + w)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * ((cy - h) / HEIGHT)
        y1 = 1.0 - 2.0 * ((cy + h) / HEIGHT)
        return x0, y0, x1, y1

    r = g = b = 1.0
    depth = 0.0  # helt foran

    # horisontal strek
    x0, y0, x1, y1 = rect_ndc(HALF_W, HALF_H, size_px, thickness_px // 2)
    verts.extend(
        [
            x0,
            y0,
            0.0,
            0.0,
            r,
            g,
            b,
            depth,
            x0,
            y1,
            0.0,
            1.0,
            r,
            g,
            b,
            depth,
            x1,
            y0,
            1.0,
            0.0,
            r,
            g,
            b,
            depth,
            x1,
            y0,
            1.0,
            0.0,
            r,
            g,
            b,
            depth,
            x0,
            y1,
            0.0,
            1.0,
            r,
            g,
            b,
            depth,
            x1,
            y1,
            1.0,
            1.0,
            r,
            g,
            b,
            depth,
        ]
    )

    # vertikal strek
    x0, y0, x1, y1 = rect_ndc(HALF_W, HALF_H, thickness_px // 2, size_px)
    verts.extend(
        [
            x0,
            y0,
            0.0,
            0.0,
            r,
            g,
            b,
            depth,
            x0,
            y1,
            0.0,
            1.0,
            r,
            g,
            b,
            depth,
            x1,
            y0,
            1.0,
            0.0,
            r,
            g,
            b,
            depth,
            x1,
            y0,
            1.0,
            0.0,
            r,
            g,
            b,
            depth,
            x0,
            y1,
            0.0,
            1.0,
            r,
            g,
            b,
            depth,
            x1,
            y1,
            1.0,
            1.0,
            r,
            g,
            b,
            depth,
        ]
    )

    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_weapon_overlay(firing: bool, recoil_t: float) -> np.ndarray:
    """En enkel "pistolboks" nederst (farget quad), m/ liten recoil-bevegelse."""
    base_w, base_h = 200, 120
    x = HALF_W - base_w // 2
    y = HEIGHT - base_h - 10
    if firing:
        y += int(6 * math.sin(min(1.0, recoil_t) * math.pi))

    x0 = (2.0 * x) / WIDTH - 1.0
    x1 = (2.0 * (x + base_w)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * (y / HEIGHT)
    y1 = 1.0 - 2.0 * ((y + base_h) / HEIGHT)

    # lett gjennomsiktig mørk grå
    # vi bruker v_col for RGB, alpha kommer fra tekstur (1x1 hvit, a=1). For alpha: n.a. her.
    r, g, b = (0.12, 0.12, 0.12)
    depth = 0.0
    verts = [
        x0,
        y0,
        0.0,
        0.0,
        r,
        g,
        b,
        depth,
        x0,
        y1,
        0.0,
        1.0,
        r,
        g,
        b,
        depth,
        x1,
        y0,
        1.0,
        0.0,
        r,
        g,
        b,
        depth,
        x1,
        y0,
        1.0,
        0.0,
        r,
        g,
        b,
        depth,
        x0,
        y1,
        0.0,
        1.0,
        r,
        g,
        b,
        depth,
        x1,
        y1,
        1.0,
        1.0,
        r,
        g,
        b,
        depth,
    ]
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_minimap_quads(enemy_list: list[Enemy]) -> np.ndarray:
    """Liten GL-basert minimap øverst til venstre."""
    scale = 6
    mm_w = MAP_W * scale
    mm_h = MAP_H * scale
    pad = 10
    verts: list[float] = []

    def add_quad_px(x_px, y_px, w_px, h_px, col, depth):
        r, g, b = col
        x0 = (2.0 * x_px) / WIDTH - 1.0
        x1 = (2.0 * (x_px + w_px)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (y_px / HEIGHT)
        y1 = 1.0 - 2.0 * ((y_px + h_px) / HEIGHT)
        verts.extend(
            [
                x0,
                y0,
                0.0,
                0.0,
                r,
                g,
                b,
                depth,
                x0,
                y1,
                0.0,
                1.0,
                r,
                g,
                b,
                depth,
                x1,
                y0,
                1.0,
                0.0,
                r,
                g,
                b,
                depth,
                x1,
                y0,
                1.0,
                0.0,
                r,
                g,
                b,
                depth,
                x0,
                y1,
                0.0,
                1.0,
                r,
                g,
                b,
                depth,
                x1,
                y1,
                1.0,
                1.0,
                r,
                g,
                b,
                depth,
            ]
        )

    # Bakgrunn
    add_quad_px(pad - 2, pad - 2, mm_w + 4, mm_h + 4, (0.1, 0.1, 0.1), 0.0)

    # Celler
    for y in range(MAP_H):
        for x in range(MAP_W):
            if MAP[y][x] > 0:
                col = (0.86, 0.86, 0.86)
                add_quad_px(pad + x * scale, pad + y * scale, scale - 1, scale - 1, col, 0.0)

    # Spiller
    px = int(player_x * scale)
    py = int(player_y * scale)
    add_quad_px(pad + px - 2, pad + py - 2, 4, 4, (1.0, 0.3, 0.3), 0.0)

    # Retningsstrek (en liten rektangulær "linje")
    fx = int(px + dir_x * 8)
    fy = int(py + dir_y * 8)
    # tegn som tynn boks mellom (px,py) og (fx,fy)
    # for enkelhet: bare en liten boks på enden
    add_quad_px(pad + fx - 1, pad + fy - 1, 2, 2, (1.0, 0.3, 0.3), 0.0)

    # Fiender (tydelige markører med mørk outline)
    if enemy_list:
        for e in enemy_list:
            if not e.alive:
                continue
            ex = int(e.x * scale)
            ey = int(e.y * scale)
            # outline (sort) og fyllfarge pr type
            if e.enemy_type == ENEMY_BLACK_HELMET:
                col = (1.0, 0.2, 0.2)  # hard = rød
            elif e.enemy_type == ENEMY_GREY_HELMET:
                col = (1.0, 0.6, 0.0)  # medium = oransje
            else:
                col = (1.0, 1.0, 0.2)  # lett = gul
            # lett puls ved skade (litt lysere)
            if getattr(e, "hurt_t", 0.0) > 0.0:
                col = (min(1.0, col[0] + 0.2), min(1.0, col[1] + 0.2), min(1.0, col[2] + 0.2))
            # outline først (større)
            add_quad_px(pad + ex - 4, pad + ey - 4, 9, 9, (0.0, 0.0, 0.0), 0.0)
            # fyll (lys farge)
            add_quad_px(pad + ex - 3, pad + ey - 3, 7, 7, col, 0.0)
            # indre prikk (hvit) for ekstra kontrast
            add_quad_px(pad + ex - 1, pad + ey - 1, 3, 3, (1.0, 1.0, 1.0), 0.0)

    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


# ---------- Input/fysikk ----------
def try_move(nx: float, ny: float) -> tuple[float, float]:
    if not is_wall(int(nx), int(player_y)):
        x = nx
    else:
        x = player_x
    if not is_wall(int(player_x), int(ny)):
        y = ny
    else:
        y = player_y
    return x, y


def handle_input(dt: float) -> None:
    global player_x, player_y, dir_x, dir_y, plane_x, plane_y
    keys = pygame.key.get_pressed()
    rot = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_q]:
        rot -= ROT_SPEED * dt
    if keys[pygame.K_RIGHT] or keys[pygame.K_e]:
        rot += ROT_SPEED * dt
    if rot != 0.0:
        cosr, sinr = math.cos(rot), math.sin(rot)
        ndx = dir_x * cosr - dir_y * sinr
        ndy = dir_x * sinr + dir_y * cosr
        npx = plane_x * cosr - plane_y * sinr
        npy = plane_x * sinr + plane_y * cosr
        dir_x, dir_y, plane_x, plane_y = ndx, ndy, npx, npy

    forward = 0.0
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        forward += MOVE_SPEED * dt
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        forward -= MOVE_SPEED * dt
    if forward != 0.0:
        nx = player_x + dir_x * forward
        ny = player_y + dir_y * forward
        player_x, player_y = try_move(nx, ny)

    strafe = 0.0
    if keys[pygame.K_a]:
        strafe -= STRAFE_SPEED * dt
    if keys[pygame.K_d]:
        strafe += STRAFE_SPEED * dt
    if strafe != 0.0:
        nx = player_x + (-dir_y) * strafe
        ny = player_y + (dir_x) * strafe
        player_x, player_y = try_move(nx, ny)


# ---------- Main ----------
def main() -> None:
    pygame.init()
    pygame.display.set_caption("Vibe Wolf (OpenGL)")

    # setup to make it work on mac as well...
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

    # Opprett GL-kontekst
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    gl.glViewport(0, 0, WIDTH, HEIGHT)

    clock = pygame.time.Clock()
    renderer = GLRenderer()

    # Score/tekst
    font = pygame.font.SysFont(None, 28)
    font_small = pygame.font.SysFont(None, 18)
    font_big = pygame.font.SysFont(None, 64)
    score: int = 0
    score_prev_text: str = ""
    score_tex_id: int | None = None
    score_tex_w = 0
    score_tex_h = 0
    # HP percent text cache
    hp_prev_text: str = ""
    hp_tex_id: int | None = None
    hp_tex_w = 0
    hp_tex_h = 0

    bullets: list[Bullet] = []
    firing = False
    recoil_t = 0.0

    # Ammo system
    ammo: int = 10
    pickups: list[AmmoBox] = []
    pickup_pops: list[PickupPop] = []
    # sounds (best-effort)
    pickup_snd = make_pickup_sound()
    shot_snd = make_shot_sound()
    death_snd = make_enemy_death_sound()
    player_death_snd = make_player_death_sound()
    injured_snd = load_sound_asset("injured.wav", "injured.ogg")
    injured_snd_cooldown = 0.0
    # ammo HUD cache
    ammo_prev_text: str = ""
    ammo_tex_id: int | None = None
    ammo_tex_w = 0
    ammo_tex_h = 0

    # Player health
    player_hp: float = float(PLAYER_MAX_HP)

    # Game over state
    game_over = False
    game_over_timer = 0.0
    go_tex_id: int | None = None
    go_tex_w = 0
    go_tex_h = 0

    # Waves: 10 waves total, increasing difficulty (easy → medium → hard)
    # Note: Coordinates are chosen to be in open tiles (non-wall) of MAP.
    enemy_waves: list[list[tuple[float, float, int]]] = [
        # Wave 0: very easy — 2 easy enemies
        [
            (16.5, 6.5, ENEMY_GREY_NO_HELMET),
            (14.5, 4.5, ENEMY_GREY_NO_HELMET),
        ],
        # Wave 1: easy — 3 easy enemies
        [
            (16.5, 6.5, ENEMY_GREY_NO_HELMET),
            (14.5, 4.5, ENEMY_GREY_NO_HELMET),
            (10.5, 2.5, ENEMY_GREY_NO_HELMET),
        ],
        # Wave 2: easy+ — 4 easy enemies
        [
            (16.5, 6.5, ENEMY_GREY_NO_HELMET),
            (14.5, 4.5, ENEMY_GREY_NO_HELMET),
            (10.5, 2.5, ENEMY_GREY_NO_HELMET),
            (4.5, 6.5, ENEMY_GREY_NO_HELMET),
        ],
        # Wave 3: transition — 3 easy + 1 medium
        [
            (8.5, 8.5, ENEMY_GREY_NO_HELMET),
            (10.5, 17.5, ENEMY_GREY_NO_HELMET),
            (14.5, 15.5, ENEMY_GREY_NO_HELMET),
            (7.5, 12.5, ENEMY_GREY_HELMET),
        ],
        # Wave 4: tougher mix — 2 easy + 2 medium
        [
            (16.5, 6.5, ENEMY_GREY_HELMET),
            (8.5, 8.5, ENEMY_GREY_HELMET),
            (14.5, 4.5, ENEMY_GREY_NO_HELMET),
            (4.5, 6.5, ENEMY_GREY_NO_HELMET),
        ],
        # Wave 5: medium — 3 medium enemies
        [
            (7.5, 12.5, ENEMY_GREY_HELMET),
            (15.5, 12.5, ENEMY_GREY_HELMET),
            (9.5, 10.5, ENEMY_GREY_HELMET),
        ],
        # Wave 6: ramping up — 2 medium + 1 hard
        [
            (5.5, 8.5, ENEMY_GREY_HELMET),
            (6.5, 10.5, ENEMY_GREY_HELMET),
            (16.5, 6.5, ENEMY_BLACK_HELMET),
        ],
        # Wave 7: mixed heavy — 2 hard + 2 medium
        [
            (16.5, 6.5, ENEMY_BLACK_HELMET),
            (15.5, 12.5, ENEMY_BLACK_HELMET),
            (14.5, 15.5, ENEMY_GREY_HELMET),
            (8.5, 14.5, ENEMY_GREY_HELMET),
        ],
        # Wave 8: hard — 3 hard enemies
        [
            (16.5, 6.5, ENEMY_BLACK_HELMET),
            (15.5, 12.5, ENEMY_BLACK_HELMET),
            (12.5, 8.5, ENEMY_BLACK_HELMET),
        ],
        # Wave 9: very hard — 3 hard + 3 medium
        [
            (16.5, 6.5, ENEMY_BLACK_HELMET),
            (15.5, 12.5, ENEMY_BLACK_HELMET),
            (12.5, 8.5, ENEMY_BLACK_HELMET),
            (14.5, 15.5, ENEMY_GREY_HELMET),
            (7.5, 12.5, ENEMY_GREY_HELMET),
            (10.5, 17.5, ENEMY_GREY_HELMET),
        ],
    ]
    current_wave = 0
    enemies: list[Enemy] = []

    def spawn_wave(idx: int) -> None:
        enemies.clear()
        for x, y, t in enemy_waves[idx]:
            enemies.append(Enemy(x, y, t))

    spawn_wave(current_wave)
    next_wave_delay = 0.0

    # Mus-capture (synlig cursor + crosshair)
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(True)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:  # eller en annen knapp
                    grab = not pygame.event.get_grab()
                    pygame.event.set_grab(grab)
                    pygame.mouse.set_visible(not grab)
                    print("Mouse grab:", grab)
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE and ammo > 0:
                    bx = player_x + dir_x * 0.4
                    by = player_y + dir_y * 0.4
                    bvx = dir_x * 10.0
                    bvy = dir_y * 10.0
                    bullets.append(Bullet(bx, by, bvx, bvy))
                    firing = True
                    recoil_t = 0.0
                    ammo -= 1
                    try:
                        if shot_snd is not None:
                            shot_snd.play()
                    except Exception:
                        pass
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if ammo > 0:
                    bx = player_x + dir_x * 0.4
                    by = player_y + dir_y * 0.4
                    bvx = dir_x * 10.0
                    bvy = dir_y * 10.0
                    bullets.append(Bullet(bx, by, bvx, bvy))
                    firing = True
                    recoil_t = 0.0
                    ammo -= 1
                    try:
                        if shot_snd is not None:
                            shot_snd.play()
                    except Exception:
                        pass

        handle_input(dt)

        # Oppdater bullets
        for b in bullets:
            b.update(dt)
            if not b.alive:
                continue
            for e in enemies:
                if not e.alive:
                    continue
                dx = e.x - b.x
                dy = e.y - b.y
                if dx * dx + dy * dy <= (e.radius * e.radius):
                    died = e.take_damage(1)  # deal 1 damage
                    if died:
                        score += ENEMY_SCORE.get(e.enemy_type, 100)
                        # invalider score-tekst slik at den oppdateres
                        score_prev_text = ""
                        # death sound
                        try:
                            if death_snd is not None:
                                death_snd.play()
                        except Exception:
                            pass
                        # 1 av 5 dropper ammo (20%)
                        if random.random() < 0.2:
                            # Harder enemies drop more ammo on average
                            if e.enemy_type == ENEMY_BLACK_HELMET:
                                lo, hi = 4, 6
                            elif e.enemy_type == ENEMY_GREY_HELMET:
                                lo, hi = 3, 5
                            else:
                                lo, hi = 2, 4
                            amt = random.randint(lo, hi)
                            pickups.append(AmmoBox(e.x, e.y, amt))
                    b.alive = False  # kula forbrukes
                    break
        bullets = [b for b in bullets if b.alive]

        # Oppdater fiender
        for e in enemies:
            e.update(dt)

        # Plukk opp ammo ved nærkontakt
        for p in pickups:
            if not p.alive:
                continue
            dx = p.x - player_x
            dy = p.y - player_y
            if dx * dx + dy * dy <= (0.5 * 0.5):
                ammo += p.amount
                p.alive = False
                # visual pop
                pickup_pops.append(PickupPop(p.x, p.y))
                # audio
                try:
                    if pickup_snd is not None:
                        pickup_snd.play()
                except Exception:
                    pass
        pickups = [p for p in pickups if p.alive]

        # Update pickup pops
        for fx in pickup_pops:
            fx.update(dt)
        pickup_pops = [fx for fx in pickup_pops if fx.alive]

        # Spiller tar kontakt-skade om fiender er nærme
        dps_total = 0.0
        for e in enemies:
            if not e.alive:
                continue
            dx = e.x - player_x
            dy = e.y - player_y
            if dx * dx + dy * dy <= (PLAYER_CONTACT_RADIUS * PLAYER_CONTACT_RADIUS):
                dps_total += PLAYER_CONTACT_DPS
        if dps_total > 0.0 and player_hp > 0.0:
            # play injured sound with a small cooldown to avoid spam
            if injured_snd_cooldown <= 0.0:
                try:
                    if injured_snd is not None:
                        injured_snd.play()
                except Exception:
                    pass
                injured_snd_cooldown = 0.4
            player_hp = max(0.0, player_hp - dps_total * dt)
        # update injured sound cooldown
        if injured_snd_cooldown > 0.0:
            injured_snd_cooldown = max(0.0, injured_snd_cooldown - dt)

        # Game over trigger
        if (not game_over) and player_hp <= 0.0:
            game_over = True
            game_over_timer = 2.0
            # Player death sound
            try:
                if player_death_snd is not None:
                    player_death_snd.play()
            except Exception:
                pass
            # Create texture once
            try:
                surf = font_big.render("GAME OVER", True, (255, 255, 255))
                go_tex_w, go_tex_h = surf.get_width(), surf.get_height()
                go_tex_id = surface_to_texture(surf)
            except Exception:
                go_tex_id = None

        # ---------- Render ----------
        gl.glViewport(0, 0, WIDTH, HEIGHT)
        gl.glClearColor(0.05, 0.07, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Bakgrunn (himmel/gulv)
        bg = build_fullscreen_background()
        renderer.draw_arrays(bg, renderer.white_tex, use_tex=False)

        # Vegger (batch pr. tex_id)
        batches_lists = cast_and_build_wall_batches()
        for tid, verts_list in batches_lists.items():
            if tid not in renderer.textures:
                continue
            if not verts_list:
                continue
            arr = np.asarray(verts_list, dtype=np.float32).reshape((-1, 8))
            renderer.draw_arrays(arr, renderer.textures[tid], use_tex=True)

        # Sprites (kuler)
        spr = build_sprites_batch(bullets)
        if spr.size:
            renderer.draw_arrays(spr, renderer.textures[99], use_tex=True)

        # Pickups (ammo boxes)
        pb = build_pickups_batch(pickups)
        if pb.size:
            renderer.draw_arrays(pb, renderer.textures[150], use_tex=True)

        # Pickup pop effects (render using ammo box texture, small & fading)
        pops_batch = build_pickup_pops_batch(pickup_pops)
        if pops_batch.size:
            renderer.draw_arrays(pops_batch, renderer.textures[150], use_tex=True)

        # Enemies (billboards) - render by type for different textures
        for enemy_type in [ENEMY_BLACK_HELMET, ENEMY_GREY_HELMET, ENEMY_GREY_NO_HELMET]:
            enemies_batch = build_enemies_batch_by_type(enemies, enemy_type)
            if enemies_batch.size:
                texture_id = 200 + enemy_type  # 200, 201, 202
                renderer.draw_arrays(enemies_batch, renderer.textures[texture_id], use_tex=True)

        # Crosshair
        cross = build_crosshair_quads(8, 2)
        renderer.draw_arrays(cross, renderer.white_tex, use_tex=False)

        # Weapon overlay
        if firing:
            recoil_t += dt
            if recoil_t > 0.15:
                firing = False
        overlay = build_weapon_overlay(firing, recoil_t)
        renderer.draw_arrays(overlay, renderer.white_tex, use_tex=False)

        # Minimap
        mm = build_minimap_quads(enemies)
        renderer.draw_arrays(mm, renderer.white_tex, use_tex=False)

    # Health bar (øverst, midtstilt)
        # bakgrunn + fyll som avhenger av HP
        ratio = 0.0 if PLAYER_MAX_HP <= 0 else float(player_hp) / float(PLAYER_MAX_HP)
        ratio = max(0.0, min(1.0, ratio))
        bar_w = 300
        bar_h = 12
        pad_y = 8
        x_px = HALF_W - bar_w // 2
        y_px = pad_y
        # farger
        bg_col = (0.15, 0.15, 0.15)
        # Lerp rød->grønn basert på ratio
        red = (0.9, 0.25, 0.25)
        green = (0.25, 0.9, 0.35)
        fill_col = (
            red[0] + (green[0] - red[0]) * ratio,
            red[1] + (green[1] - red[1]) * ratio,
            red[2] + (green[2] - red[2]) * ratio,
        )
        # bakgrunn quad
        x0 = (2.0 * x_px) / WIDTH - 1.0
        x1 = (2.0 * (x_px + bar_w)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (y_px / HEIGHT)
        y1 = 1.0 - 2.0 * ((y_px + bar_h) / HEIGHT)
        depth = 0.0
        verts_hud: list[float] = []

        def add_rect(x0f, y0f, x1f, y1f, col):
            r, g, b = col
            verts_hud.extend(
                [
                    x0f,
                    y0f,
                    0.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                    x0f,
                    y1f,
                    0.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    x1f,
                    y0f,
                    1.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                    x1f,
                    y0f,
                    1.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                    x0f,
                    y1f,
                    0.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    x1f,
                    y1f,
                    1.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                ]
            )

        add_rect(x0, y0, x1, y1, bg_col)
        # fyll bredde
        fill_w = int(bar_w * ratio)
        if fill_w > 0:
            fx1 = (2.0 * (x_px + fill_w)) / WIDTH - 1.0
            add_rect(x0, y0, fx1, y1, fill_col)
        if verts_hud:
            renderer.draw_arrays(
                np.asarray(verts_hud, dtype=np.float32).reshape((-1, 8)),
                renderer.white_tex,
                use_tex=False,
            )

        # HP prosent-tekst, sentrert i baren
        hp_pct = int(round(ratio * 100.0))
        hp_text = f"{hp_pct}%"
        if hp_text != hp_prev_text:
            if hp_tex_id is not None:
                try:
                    gl.glDeleteTextures([hp_tex_id])
                except Exception:
                    pass
                hp_tex_id = None
            hp_surf = font_small.render(hp_text, True, (255, 255, 255))
            hp_tex_w, hp_tex_h = hp_surf.get_width(), hp_surf.get_height()
            hp_tex_id = surface_to_texture(hp_surf)
            hp_prev_text = hp_text

        if hp_tex_id is not None:
            tx = HALF_W - hp_tex_w // 2
            ty = y_px + (bar_h - hp_tex_h) // 2
            x0t = (2.0 * tx) / WIDTH - 1.0
            x1t = (2.0 * (tx + hp_tex_w)) / WIDTH - 1.0
            y0t = 1.0 - 2.0 * (ty / HEIGHT)
            y1t = 1.0 - 2.0 * ((ty + hp_tex_h) / HEIGHT)
            r = g = b = 1.0
            depth = 0.0
            hp_verts = np.asarray(
                [
                    x0t,
                    y0t,
                    0.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    x0t,
                    y1t,
                    0.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                    x1t,
                    y0t,
                    1.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    x1t,
                    y0t,
                    1.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    x0t,
                    y1t,
                    0.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                    x1t,
                    y1t,
                    1.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                ],
                dtype=np.float32,
            ).reshape((-1, 8))
            renderer.draw_arrays(hp_verts, hp_tex_id, use_tex=True)

        # Ammo-tekst (ved siden av HP-bar)
        ammo_text = f"Ammo: {ammo}"
        if ammo_text != ammo_prev_text:
            if ammo_tex_id is not None:
                try:
                    gl.glDeleteTextures([ammo_tex_id])
                except Exception:
                    pass
                ammo_tex_id = None
            a_surf = font.render(ammo_text, True, (255, 255, 255))
            ammo_tex_w, ammo_tex_h = a_surf.get_width(), a_surf.get_height()
            ammo_tex_id = surface_to_texture(a_surf)
            ammo_prev_text = ammo_text

        if ammo_tex_id is not None:
            pad_x = 10
            # place to the right of HP bar
            x_px2 = HALF_W + 300 // 2 + pad_x
            y_px2 = pad_y
            x0a = (2.0 * x_px2) / WIDTH - 1.0
            x1a = (2.0 * (x_px2 + ammo_tex_w)) / WIDTH - 1.0
            y0a = 1.0 - 2.0 * (y_px2 / HEIGHT)
            y1a = 1.0 - 2.0 * ((y_px2 + ammo_tex_h) / HEIGHT)
            r = g = b = 1.0
            depth = 0.0
            ammo_verts = np.asarray(
                [
                    x0a, y0a, 0.0, 1.0, r, g, b, depth,
                    x0a, y1a, 0.0, 0.0, r, g, b, depth,
                    x1a, y0a, 1.0, 1.0, r, g, b, depth,
                    x1a, y0a, 1.0, 1.0, r, g, b, depth,
                    x0a, y1a, 0.0, 0.0, r, g, b, depth,
                    x1a, y1a, 1.0, 0.0, r, g, b, depth,
                ],
                dtype=np.float32,
            ).reshape((-1, 8))
            renderer.draw_arrays(ammo_verts, ammo_tex_id, use_tex=True)

        # Score-tekst (øverst til høyre)
        score_text = f"Score: {score}"
        if score_text != score_prev_text:
            # oppdater tekstur (frigjør gammel)
            if score_tex_id is not None:
                try:
                    gl.glDeleteTextures([score_tex_id])
                except Exception:
                    pass
                score_tex_id = None
            surf = font.render(score_text, True, (255, 255, 255))
            score_tex_w, score_tex_h = surf.get_width(), surf.get_height()
            score_tex_id = surface_to_texture(surf)
            score_prev_text = score_text

        if score_tex_id is not None:
            pad = 10
            x_px = WIDTH - pad - score_tex_w
            y_px = pad
            # bygg et enkelt quad i px-koordinater
            x0 = (2.0 * x_px) / WIDTH - 1.0
            x1 = (2.0 * (x_px + score_tex_w)) / WIDTH - 1.0
            y0 = 1.0 - 2.0 * (y_px / HEIGHT)
            y1 = 1.0 - 2.0 * ((y_px + score_tex_h) / HEIGHT)
            r = g = b = 1.0
            depth = 0.0
            verts = np.asarray(
                [
                    # top-left
                    x0,
                    y0,
                    0.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    # bottom-left
                    x0,
                    y1,
                    0.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                    # top-right
                    x1,
                    y0,
                    1.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    # top-right
                    x1,
                    y0,
                    1.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    # bottom-left
                    x0,
                    y1,
                    0.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                    # bottom-right
                    x1,
                    y1,
                    1.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                ],
                dtype=np.float32,
            ).reshape((-1, 8))
            renderer.draw_arrays(verts, score_tex_id, use_tex=True)

        # Game over overlay (sentrert)
        if game_over and go_tex_id is not None:
            tx = HALF_W - go_tex_w // 2
            ty = HALF_H - go_tex_h // 2
            x0 = (2.0 * tx) / WIDTH - 1.0
            x1 = (2.0 * (tx + go_tex_w)) / WIDTH - 1.0
            y0 = 1.0 - 2.0 * (ty / HEIGHT)
            y1 = 1.0 - 2.0 * ((ty + go_tex_h) / HEIGHT)
            r = g = b = 1.0
            depth = 0.0
            go_verts = np.asarray(
                [
                    x0,
                    y0,
                    0.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    x0,
                    y1,
                    0.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                    x1,
                    y0,
                    1.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    x1,
                    y0,
                    1.0,
                    1.0,
                    r,
                    g,
                    b,
                    depth,
                    x0,
                    y1,
                    0.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                    x1,
                    y1,
                    1.0,
                    0.0,
                    r,
                    g,
                    b,
                    depth,
                ],
                dtype=np.float32,
            ).reshape((-1, 8))
            renderer.draw_arrays(go_verts, go_tex_id, use_tex=True)

        pygame.display.flip()

        # Wave progression (when all enemies are dead)
        if not game_over:
            if all((not e.alive) for e in enemies):
                if next_wave_delay <= 0.0 and current_wave + 1 < len(enemy_waves):
                    next_wave_delay = 1.0  # small pause before next wave
                if next_wave_delay > 0.0:
                    next_wave_delay -= dt
                    if next_wave_delay <= 0.0:
                        current_wave += 1
                        spawn_wave(current_wave)

        # Game over countdown -> exit
        if game_over:
            game_over_timer -= dt
            if game_over_timer <= 0.0:
                running = False

    pygame.quit()

    # Frigi dynamisk tekstur ved avslutning
    try:
        if score_tex_id is not None:
            gl.glDeleteTextures([score_tex_id])
        if hp_tex_id is not None:
            gl.glDeleteTextures([hp_tex_id])
        if ammo_tex_id is not None:
            gl.glDeleteTextures([ammo_tex_id])
        if go_tex_id is not None:
            gl.glDeleteTextures([go_tex_id])
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
