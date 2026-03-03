import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import pygame

# -----------------------------
# Display + Radar scaling
# -----------------------------
W, H = 1200, 720
PANEL_W = 360
RADAR_W = W - PANEL_W
CENTER = (RADAR_W // 2, H // 2)
RADAR_RADIUS = 300

FPS = 60

# Scale: radar radius == this many "km" (for labels only)
RADAR_RANGE_KM = 500.0
PX_PER_KM = RADAR_RADIUS / RADAR_RANGE_KM

def km_to_px(km: float) -> float:
    return km * PX_PER_KM

def px_to_km(px: float) -> float:
    return px / PX_PER_KM

# -----------------------------
# Gameplay tuning
# -----------------------------
MAX_CONTACTS = 14
SPAWN_GAP = (0.6, 1.6)

# Time-to-impact (from max range to center), in seconds (tuned for fun + readability)
TTI = {
    "HYPERSONIC": (5.5, 7.0),
    "BALLISTIC":  (12.0, 16.0),
    "CRUISE":     (22.0, 30.0),
    "AIR":        (45.0, 65.0),
}

# Spawn weights
SPAWN_WEIGHTS = {"AIR": 0.38, "CRUISE": 0.30, "BALLISTIC": 0.22, "HYPERSONIC": 0.10}

# Classifier confidence ranges
CONF_RANGE = {
    "AIR": (0.55, 0.90),
    "CRUISE": (0.65, 0.92),
    "BALLISTIC": (0.70, 0.95),
    "HYPERSONIC": (0.70, 0.96),
}

# IFF (AIR only)
AIR_FRIEND_PROB = 0.35
IFF_NO_REPLY_PROB = 0.12

# Interceptors (midcourse-ish)
INTERCEPTOR_SPEED_KM_S = 18.0
INTERCEPTOR_SPEED = km_to_px(INTERCEPTOR_SPEED_KM_S)  # px/s
INTERCEPTOR_COOLDOWN = 0.55

# Magazine + reload
MAG_SIZE = 10
RESERVE_AMMO = 24
RELOAD_TIME = 2.2  # seconds
MANUAL_RELOAD_KEY = pygame.K_r

# Salvos
SALVO_SPREAD_SEC = 0.08  # slight staggering so it "feels" like a salvo

# Kill probability model (game-tuned)
BASE_PK = {"AIR": 0.75, "CRUISE": 0.70, "BALLISTIC": 0.55, "HYPERSONIC": 0.42}
LATE_SHOT_PENALTY_PER_SEC = 0.06

# Battery health (damage per leak by kind)
BASE_HP_MAX = 100
LEAK_DAMAGE = {"AIR": 10, "CRUISE": 18, "BALLISTIC": 26, "HYPERSONIC": 34}

# C-RAM
CRAM_RANGE_KM = 35.0
CRAM_RANGE = km_to_px(CRAM_RANGE_KM)
CRAM_COOLDOWN = 0.14     # burst cadence
CRAM_BURST_SHOTS = 18    # shots per press
CRAM_AMMO_MAX = 260
CRAM_RELOAD_TIME = 3.2
CRAM_HIT_CHANCE_BASE = 0.18  # per shot base (tuned)
CRAM_KIND_MULT = {"AIR": 0.30, "CRUISE": 0.85, "BALLISTIC": 0.55, "HYPERSONIC": 0.38}

# -----------------------------
# Helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def vec_len(x, y):
    return math.hypot(x, y)

def angle_to(dx, dy):
    return math.atan2(dy, dx)

def wrap_angle(a):
    a %= math.tau
    if a < 0:
        a += math.tau
    return a

def polar_from_center(x, y):
    dx = x - CENTER[0]
    dy = y - CENTER[1]
    r = vec_len(dx, dy)
    th = wrap_angle(angle_to(dx, dy))
    return r, th

def xy_from_polar(r, th):
    return (CENTER[0] + math.cos(th) * r, CENTER[1] + math.sin(th) * r)

def within_radar(x, y):
    return vec_len(x - CENTER[0], y - CENTER[1]) <= RADAR_RADIUS

def choose_weighted(d: dict) -> str:
    r = random.random()
    acc = 0.0
    for k, w in d.items():
        acc += w
        if r <= acc:
            return k
    return list(d.keys())[-1]

def intercept_point(shooter: Tuple[float, float], target_pos: Tuple[float, float], target_vel: Tuple[float, float], missile_speed: float):
    sx, sy = shooter
    tx, ty = target_pos
    tvx, tvy = target_vel
    rx, ry = tx - sx, ty - sy

    a = (tvx * tvx + tvy * tvy) - missile_speed * missile_speed
    b = 2 * (rx * tvx + ry * tvy)
    c = rx * rx + ry * ry

    if abs(a) < 1e-6:
        if abs(b) < 1e-6:
            return target_pos
        t = max(0.0, -c / b)
        return (tx + tvx * t, ty + tvy * t)

    disc = b * b - 4 * a * c
    if disc < 0:
        return target_pos

    s = math.sqrt(disc)
    t1 = (-b + s) / (2 * a)
    t2 = (-b - s) / (2 * a)
    ts = [t for t in (t1, t2) if t > 0]
    if not ts:
        return target_pos
    t = min(ts)
    return (tx + tvx * t, ty + tvy * t)

# -----------------------------
# Entities
# -----------------------------
@dataclass
class Contact:
    cid: int
    kind: str
    x: float
    y: float
    vx: float
    vy: float
    spawned_at: float

    confidence: float = 0.75
    iff_state: str = "N/A"   # N/A / UNKNOWN / FRIEND / FOE / NO_REPLY
    iff_last: float = -999.0
    destroyed: bool = False
    leaked: bool = False

    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def distance_km(self) -> float:
        r, _ = polar_from_center(self.x, self.y)
        return px_to_km(r)

    def time_to_impact(self) -> float:
        dx = CENTER[0] - self.x
        dy = CENTER[1] - self.y
        dist = vec_len(dx, dy)
        if dist < 1e-6:
            return 0.0
        ux, uy = dx / dist, dy / dist
        closing = self.vx * ux + self.vy * uy
        if closing <= 1e-3:
            return 999.0
        return dist / closing

    def impacted(self) -> bool:
        return vec_len(self.x - CENTER[0], self.y - CENTER[1]) <= 10

@dataclass
class Interceptor:
    x: float
    y: float
    tx: float
    ty: float
    speed: float
    alive: bool = True
    launched_at: float = 0.0

    def update(self, dt: float):
        dx = self.tx - self.x
        dy = self.ty - self.y
        d = vec_len(dx, dy)
        if d < 7:
            self.alive = False
            return
        ux, uy = dx / d, dy / d
        step = self.speed * dt
        self.x += ux * step
        self.y += uy * step

# -----------------------------
# Game state
# -----------------------------
class Game:
    def __init__(self):
        self.contacts: List[Contact] = []
        self.interceptors: List[Interceptor] = []
        self.next_id = 1
        self.locked_id: Optional[int] = None

        self.next_spawn = time.time() + random.uniform(*SPAWN_GAP)

        self.score = 0
        self.game_over = False

        # Battery
        self.base_hp = BASE_HP_MAX

        # Magazine + reserve
        self.mag = MAG_SIZE
        self.reserve = RESERVE_AMMO
        self.reloading_until = -999.0
        self.last_shot = -999.0

        # C-RAM
        self.cram_ammo = CRAM_AMMO_MAX
        self.cram_reloading_until = -999.0
        self.cram_last_shot = -999.0

        self.message = ""
        self.message_until = 0.0

        # “sweep” purely for vibes
        self.sweep = 0.0
        self.sweep_speed = 1.25

        # queued salvo shots (launch times)
        self.salvo_queue: List[Tuple[float, int]] = []  # (launch_time, cid)

    def set_msg(self, s: str, dur: float = 1.4):
        self.message = s
        self.message_until = time.time() + dur

    def spawn_contact(self):
        if len([c for c in self.contacts if not c.destroyed]) >= MAX_CONTACTS:
            return

        kind = choose_weighted(SPAWN_WEIGHTS)

        th = random.uniform(0, math.tau)
        r = RADAR_RADIUS + 10
        x, y = xy_from_polar(r, th)

        tti = random.uniform(*TTI[kind])
        to_center = angle_to(CENTER[0] - x, CENTER[1] - y) + random.uniform(-0.12, 0.12)
        speed = RADAR_RADIUS / tti

        vx = math.cos(to_center) * speed
        vy = math.sin(to_center) * speed

        conf = random.uniform(*CONF_RANGE[kind])
        iff_state = "UNKNOWN" if kind == "AIR" else "N/A"

        c = Contact(
            cid=self.next_id,
            kind=kind,
            x=x, y=y,
            vx=vx, vy=vy,
            spawned_at=time.time(),
            confidence=conf,
            iff_state=iff_state
        )
        self.contacts.append(c)
        self.next_id += 1

    def find_contact(self, cid: Optional[int]) -> Optional[Contact]:
        if cid is None:
            return None
        for c in self.contacts:
            if c.cid == cid and not c.destroyed:
                return c
        return None

    def lock_nearest(self, mx, my):
        best = None
        best_d = 1e9
        for c in self.contacts:
            if c.destroyed:
                continue
            if not within_radar(c.x, c.y):
                continue
            d = vec_len(c.x - mx, c.y - my)
            if d < best_d:
                best_d = d
                best = c.cid
        self.locked_id = best

    def cycle_threat(self):
     living = [c for c in self.contacts if not c.destroyed]
     if not living:
        self.locked_id = None
        return

     kind_weight = {"HYPERSONIC": 0, "BALLISTIC": 1, "CRUISE": 2, "AIR": 3}
     ordered = sorted(living, key=lambda c: (c.time_to_impact(), kind_weight[c.kind]))

     if self.locked_id is None or self.locked_id not in [c.cid for c in ordered]:
        self.locked_id = ordered[0].cid
        return

     tids = [c.cid for c in ordered]
     idx = tids.index(self.locked_id)
     self.locked_id = tids[(idx + 1) % len(tids)]

    # ----------------
    # IFF
    # ----------------
    def iff_interrogate(self):
        c = self.find_contact(self.locked_id)
        if c is None:
            return
        if c.kind != "AIR":
            self.set_msg("IFF only relevant for AIR tracks.")
            return
        c.iff_last = time.time()

        r = random.random()
        if r < IFF_NO_REPLY_PROB:
            c.iff_state = "NO_REPLY"
            self.set_msg(f"T{c.cid:02d} IFF: NO REPLY")
        else:
            if random.random() < AIR_FRIEND_PROB:
                c.iff_state = "FRIEND"
                self.set_msg(f"T{c.cid:02d} IFF: FRIEND")
            else:
                c.iff_state = "FOE"
                self.set_msg(f"T{c.cid:02d} IFF: FOE")

    # ----------------
    # Reload
    # ----------------
    def start_reload(self):
        now = time.time()
        if now < self.reloading_until:
            return
        if self.mag == MAG_SIZE:
            self.set_msg("Magazine already full.")
            return
        if self.reserve <= 0:
            self.set_msg("No reserve ammo left.")
            return
        self.reloading_until = now + RELOAD_TIME
        self.set_msg(f"RELOADING... ({RELOAD_TIME:.1f}s)")

    def finish_reload_if_ready(self):
        now = time.time()
        if now < self.reloading_until:
            return
        if self.reloading_until < 0:
            return
        # finalize reload
        needed = MAG_SIZE - self.mag
        take = min(needed, self.reserve)
        self.mag += take
        self.reserve -= take
        self.reloading_until = -999.0
        self.set_msg(f"Reload complete. MAG {self.mag}/{MAG_SIZE}")

    # ----------------
    # Fire / Salvos
    # ----------------
    def can_fire(self) -> bool:
        now = time.time()
        if self.game_over:
            return False
        if now < self.reloading_until:
            return False
        if (now - self.last_shot) < INTERCEPTOR_COOLDOWN:
            return False
        if self.mag <= 0:
            return False
        return True

    def queue_salvo(self, n: int):
        c = self.find_contact(self.locked_id)
        if c is None:
            self.set_msg("No locked target.")
            return

        # ROE: must IFF AIR before firing (no override here)
        if c.kind == "AIR" and c.iff_state != "FOE":
            self.set_msg("ROE: IFF AIR first (I) before launch.")
            return

        # if magazine empty, suggest reload
        if self.mag <= 0:
            self.set_msg("MAG EMPTY. Press R to reload.")
            return

        # queue up to n shots, limited by available mag
        n = min(n, self.mag)
        if n <= 0:
            return

        base_t = time.time()
        for i in range(n):
            self.salvo_queue.append((base_t + i * SALVO_SPREAD_SEC, c.cid))

        self.set_msg(f"SALVO queued: {n}x at T{c.cid:02d}", 1.2)

    def launch_if_due(self):
        # launch queued salvo shots when their time comes (honors cooldown lightly)
        if not self.salvo_queue:
            return
        now = time.time()
        # sort by time
        self.salvo_queue.sort(key=lambda x: x[0])

        launched_any = False
        while self.salvo_queue and self.salvo_queue[0][0] <= now:
            _, cid = self.salvo_queue.pop(0)

            # target still alive?
            c = self.find_contact(cid)
            if c is None:
                continue

            # check basic constraints
            if not self.can_fire():
                # if can't fire right now, reinsert with slight delay (don't spam)
                self.salvo_queue.insert(0, (now + 0.12, cid))
                break

            lead = intercept_point((CENTER[0], CENTER[1]), (c.x, c.y), (c.vx, c.vy), INTERCEPTOR_SPEED)
            self.interceptors.append(Interceptor(CENTER[0], CENTER[1], lead[0], lead[1], INTERCEPTOR_SPEED, launched_at=now))
            self.mag -= 1
            self.last_shot = now
            launched_any = True

            if self.mag == 0:
                self.set_msg("MAG EMPTY. Press R to reload.", 1.2)
                break

        if launched_any:
            pass

    # ----------------
    # Intercept model
    # ----------------
    def resolve_intercept(self, c: Contact) -> bool:
        tti = c.time_to_impact()
        late_pen = max(0.0, 6.0 - tti) * LATE_SHOT_PENALTY_PER_SEC
        pk = BASE_PK[c.kind] - late_pen
        # salvo effect: later missiles add diminishing boost; handled by multiple trials anyway
        return random.random() < clamp(pk, 0.08, 0.92)

    # ----------------
    # C-RAM
    # ----------------
    def cram_ready(self) -> bool:
        now = time.time()
        if now < self.cram_reloading_until:
            return False
        return (now - self.cram_last_shot) >= CRAM_COOLDOWN and self.cram_ammo > 0 and not self.game_over

    def cram_reload(self):
        now = time.time()
        if now < self.cram_reloading_until:
            return
        self.cram_reloading_until = now + CRAM_RELOAD_TIME
        self.set_msg(f"C-RAM RELOADING... ({CRAM_RELOAD_TIME:.1f}s)", 1.4)

    def cram_finish_reload_if_ready(self):
        now = time.time()
        if self.cram_reloading_until > 0 and now >= self.cram_reloading_until:
            self.cram_ammo = CRAM_AMMO_MAX
            self.cram_reloading_until = -999.0
            self.set_msg("C-RAM reload complete.", 1.1)

    def do_cram_burst(self):
        # Fires a burst at the most urgent in-range target
        if not self.cram_ready():
            return

        # pick closest-to-impact contact within CRAM range
        candidates = []
        for c in self.contacts:
            if c.destroyed:
                continue
            r_px, _ = polar_from_center(c.x, c.y)
            if r_px <= CRAM_RANGE:
                candidates.append(c)

        if not candidates:
            self.set_msg("C-RAM: No targets in range.", 1.0)
            return

        candidates.sort(key=lambda c: c.time_to_impact())
        target = candidates[0]

        shots = min(CRAM_BURST_SHOTS, self.cram_ammo)
        if shots <= 0:
            self.cram_reload()
            return

        # Each shot has small chance; cruise is easier, hypersonic is hard.
        # Also slightly better the closer it is.
        r_px, _ = polar_from_center(target.x, target.y)
        proximity = clamp(1.0 - (r_px / CRAM_RANGE), 0.0, 1.0)

        p_hit = CRAM_HIT_CHANCE_BASE * CRAM_KIND_MULT[target.kind] * (0.55 + 0.85 * proximity)
        p_hit = clamp(p_hit, 0.02, 0.55)

        killed = False
        for _ in range(shots):
            if random.random() < p_hit:
                killed = True
                break

        self.cram_ammo -= shots
        self.cram_last_shot = time.time()

        if killed:
            target.destroyed = True
            self.score += {"AIR": 25, "CRUISE": 70, "BALLISTIC": 110, "HYPERSONIC": 170}[target.kind]
            self.set_msg(f"C-RAM KILL on T{target.cid:02d} ({target.kind})", 1.2)
        else:
            self.set_msg(f"C-RAM burst ineffective on T{target.cid:02d}", 0.9)

        if self.cram_ammo <= 0:
            self.cram_reload()

    # ----------------
    # Update
    # ----------------
    def update(self, dt: float):
        if self.game_over:
            return

        now = time.time()

        # finalize reloads if ready
        self.finish_reload_if_ready()
        self.cram_finish_reload_if_ready()

        # spawn
        if now >= self.next_spawn:
            self.spawn_contact()
            self.next_spawn = now + random.uniform(*SPAWN_GAP)

        # sweep vibe
        self.sweep = wrap_angle(self.sweep + self.sweep_speed * dt)

        # update contacts
        for c in self.contacts:
            if c.destroyed:
                continue
            c.update(dt)

        # impacts -> base damage
        for c in self.contacts:
            if c.destroyed:
                continue
            if c.impacted():
                c.destroyed = True
                c.leaked = True
                dmg = LEAK_DAMAGE[c.kind]
                self.base_hp -= dmg
                self.base_hp = max(0, self.base_hp)
                self.set_msg(f"IMPACT! T{c.cid:02d} {c.kind} leaked. -{dmg} HP", 1.6)
                if self.base_hp <= 0:
                    self.game_over = True
                    self.set_msg("BASE DESTROYED. GAME OVER.", 10)

        # queued salvo launches
        self.launch_if_due()

        # update interceptors
        for m in self.interceptors:
            if not m.alive:
                continue
            m.update(dt)

        # check interceptors near contacts
        for m in self.interceptors:
            if not m.alive:
                continue
            for c in self.contacts:
                if c.destroyed:
                    continue
                if vec_len(m.x - c.x, m.y - c.y) < 14:
                    m.alive = False
                    if self.resolve_intercept(c):
                        c.destroyed = True
                        self.score += {"AIR": 50, "CRUISE": 90, "BALLISTIC": 140, "HYPERSONIC": 220}[c.kind]
                        self.set_msg(f"INTERCEPT SUCCESS on T{c.cid:02d} ({c.kind})", 1.2)
                    else:
                        self.set_msg(f"INTERCEPT MISS on T{c.cid:02d} ({c.kind})", 1.2)
                    break

        self.interceptors = [m for m in self.interceptors if m.alive]

        # clean old destroyed contacts a bit
        self.contacts = [c for c in self.contacts if not (c.destroyed and (time.time() - c.spawned_at) > 10)]

        # ensure lock exists
        if self.locked_id is not None and self.find_contact(self.locked_id) is None:
            self.locked_id = None

# -----------------------------
# Drawing
# -----------------------------
def draw_text(screen, font, s, x, y, color=(220, 240, 220)):
    surf = font.render(s, True, color)
    screen.blit(surf, (x, y))

def draw_bar(screen, x, y, w, h, frac, fg=(80, 240, 200), bg=(25, 35, 35), border=(60, 80, 80)):
    pygame.draw.rect(screen, bg, (x, y, w, h), border_radius=6)
    pygame.draw.rect(screen, fg, (x, y, int(w * clamp(frac, 0, 1)), h), border_radius=6)
    pygame.draw.rect(screen, border, (x, y, w, h), width=1, border_radius=6)

def draw_radar(surface, game: Game, font_small):
    surface.fill((6, 10, 10))

    pygame.draw.circle(surface, (18, 60, 50), CENTER, RADAR_RADIUS, 2)
    for frac in [0.2, 0.4, 0.6, 0.8]:
        pygame.draw.circle(surface, (12, 38, 32), CENTER, int(RADAR_RADIUS * frac), 1)

    pygame.draw.line(surface, (12, 38, 32), (CENTER[0] - RADAR_RADIUS, CENTER[1]), (CENTER[0] + RADAR_RADIUS, CENTER[1]), 1)
    pygame.draw.line(surface, (12, 38, 32), (CENTER[0], CENTER[1] - RADAR_RADIUS), (CENTER[0], CENTER[1] + RADAR_RADIUS), 1)

    sx = CENTER[0] + math.cos(game.sweep) * RADAR_RADIUS
    sy = CENTER[1] + math.sin(game.sweep) * RADAR_RADIUS
    pygame.draw.line(surface, (60, 230, 160), CENTER, (sx, sy), 2)

    pygame.draw.circle(surface, (80, 240, 200), CENTER, 8)
    pygame.draw.circle(surface, (20, 110, 95), CENTER, 20, 1)

    # C-RAM zone ring
    pygame.draw.circle(surface, (25, 55, 55), CENTER, int(CRAM_RANGE), 1)

    # contacts
    for c in game.contacts:
        if c.destroyed:
            continue
        if not within_radar(c.x, c.y):
            continue

        if c.kind == "HYPERSONIC":
            col = (255, 90, 90)
        elif c.kind == "BALLISTIC":
            col = (255, 160, 90)
        elif c.kind == "CRUISE":
            col = (255, 230, 120)
        else:
            col = (140, 200, 255)

        pygame.draw.circle(surface, col, (int(c.x), int(c.y)), 5)

        sp = vec_len(c.vx, c.vy)
        if sp > 0.1:
            ex = c.x + (c.vx / sp) * 18
            ey = c.y + (c.vy / sp) * 18
            pygame.draw.line(surface, col, (c.x, c.y), (ex, ey), 2)

        rng_km = int(c.distance_km())
        label = f"T{c.cid:02d}  {rng_km}km"
        draw_text(surface, font_small, label, int(c.x) + 10, int(c.y) - 12, (210, 230, 210))

        tti = c.time_to_impact()
        conf = int(c.confidence * 100)
        sub = f"{c.kind} {conf}%  TTI {tti:>4.1f}s"
        draw_text(surface, font_small, sub, int(c.x) + 10, int(c.y) + 6, (150, 185, 180))

        if game.locked_id == c.cid:
            pygame.draw.circle(surface, (250, 240, 120), (int(c.x), int(c.y)), 16, 2)

    # interceptors
    for m in game.interceptors:
        pygame.draw.circle(surface, (255, 240, 120), (int(m.x), int(m.y)), 3)

    # top HUD
    draw_text(
        surface, font_small,
        "AIR DEFENCE RADAR  |  LMB lock  |  TAB cycle  |  I IFF  |  SPACE launch  |  1/2/3 salvos  |  R reload  |  LCTRL C-RAM",
        14, 12
    )
    if time.time() < game.message_until:
        draw_text(surface, font_small, game.message, 14, 36, (255, 230, 160))

def draw_panel(screen, game: Game, font, font_small):
    x0 = RADAR_W
    pygame.draw.rect(screen, (10, 14, 14), (x0, 0, PANEL_W, H))
    pygame.draw.line(screen, (25, 40, 40), (x0, 0), (x0, H), 2)

    draw_text(screen, font, "BATTERY STATUS", x0 + 16, 16, (200, 240, 230))

    # HP bar
    draw_text(screen, font_small, f"Base HP: {game.base_hp}/{BASE_HP_MAX}", x0 + 16, 56, (170, 190, 190))
    draw_bar(screen, x0 + 16, 78, PANEL_W - 32, 14, game.base_hp / BASE_HP_MAX, fg=(80, 240, 200))

    # Ammo / reload status
    y = 110
    draw_text(screen, font_small, f"Interceptors MAG: {game.mag}/{MAG_SIZE}", x0 + 16, y, (170, 190, 190)); y += 20
    draw_text(screen, font_small, f"Reserve: {game.reserve}", x0 + 16, y, (170, 190, 190)); y += 20
    if time.time() < game.reloading_until:
        draw_text(screen, font_small, "Reloading...", x0 + 16, y, (255, 230, 160)); y += 20

    y += 8
    draw_text(screen, font_small, f"C-RAM Ammo: {game.cram_ammo}/{CRAM_AMMO_MAX}", x0 + 16, y, (170, 190, 190)); y += 20
    if time.time() < game.cram_reloading_until:
        draw_text(screen, font_small, "C-RAM Reloading...", x0 + 16, y, (255, 230, 160)); y += 20

    y += 18
    draw_text(screen, font, "TRACK CONTROL", x0 + 16, y, (200, 240, 230)); y += 40

    locked = game.find_contact(game.locked_id)
    if locked is None:
        draw_text(screen, font_small, "Locked: NONE", x0 + 16, y, (170, 190, 190))
        y += 30
    else:
        draw_text(screen, font_small, f"Locked: T{locked.cid:02d}", x0 + 16, y, (240, 240, 200)); y += 20
        draw_text(screen, font_small, f"Type: {locked.kind} ({int(locked.confidence*100)}%)", x0 + 16, y, (170, 190, 190)); y += 20
        draw_text(screen, font_small, f"Range: {locked.distance_km():.1f} km", x0 + 16, y, (170, 190, 190)); y += 20
        draw_text(screen, font_small, f"TTI:   {locked.time_to_impact():.1f} s", x0 + 16, y, (170, 190, 190)); y += 20

        if locked.kind == "AIR":
            draw_text(screen, font_small, f"IFF:   {locked.iff_state}", x0 + 16, y, (200, 200, 170)); y += 20
            draw_text(screen, font_small, "ROE: IFF AIR before launch", x0 + 16, y, (200, 170, 150)); y += 20
        else:
            draw_text(screen, font_small, "ROE: Intercept permitted", x0 + 16, y, (160, 190, 190)); y += 20

    y += 18
    draw_text(screen, font_small, f"Score: {game.score}", x0 + 16, y, (170, 190, 190)); y += 24

    draw_text(screen, font_small, "Threats (urgency order):", x0 + 16, y, (200, 240, 230)); y += 20

    living = [c for c in game.contacts if not c.destroyed]
    kind_weight = {"HYPERSONIC": 0, "BALLISTIC": 1, "CRUISE": 2, "AIR": 3}
    ordered = sorted(living, key=lambda c: (c.time_to_impact(), kind_weight[c.kind]))

    for c in ordered[:13]:
        tti = c.time_to_impact()
        rng = int(c.distance_km())
        mark = ">" if game.locked_id == c.cid else " "
        line = f"{mark} T{c.cid:02d}  {c.kind:<10}  {rng:>3d}km  {tti:>4.1f}s"
        col = (255, 170, 170) if c.kind == "HYPERSONIC" else (170, 190, 190)
        draw_text(screen, font_small, line, x0 + 16, y, col)
        y += 18

# -----------------------------
# Main loop
# -----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Long Range Air Defence Radar (Salvos + Health + Reload + C-RAM)")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 24)
    font_small = pygame.font.SysFont("consolas", 16)

    game = Game()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_TAB:
                    game.cycle_threat()
                if event.key == pygame.K_i:
                    game.iff_interrogate()
                if event.key == pygame.K_SPACE:
                    game.queue_salvo(1)
                if event.key == pygame.K_1:
                    game.queue_salvo(1)
                if event.key == pygame.K_2:
                    game.queue_salvo(2)
                if event.key == pygame.K_3:
                    game.queue_salvo(3)
                if event.key == MANUAL_RELOAD_KEY:
                    game.start_reload()
                if event.key == pygame.K_LCTRL:
                    game.do_cram_burst()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                if mx < RADAR_W:
                    game.lock_nearest(mx, my)

        game.update(dt)

        radar_surface = screen.subsurface((0, 0, RADAR_W, H))
        draw_radar(radar_surface, game, font_small)
        draw_panel(screen, game, font, font_small)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()