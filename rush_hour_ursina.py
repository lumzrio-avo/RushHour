import collections, math, time, random, traceback
from ursina import (
    Ursina, Entity, Text, Button, color, camera, mouse,
    window, application, scene, Vec3, Vec2,
    SmoothFollow, Animation, Sequence, Func, Wait,
    destroy, invoke, held_keys, curve
)
from ursina.shaders import basic_lighting_shader, unlit_shader
from ursina.lights import AmbientLight, DirectionalLight

BOARD_COLS, BOARD_ROWS = 6, 6
CELL_SIZE = 1.0
HORIZONTAL, VERTICAL = 0, 1
TARGET_EXIT_COL = 5
TARGET_EXIT_ROW = 2

class Direction:
    HORIZONTAL = HORIZONTAL
    VERTICAL   = VERTICAL

class Vehicle:
    def __init__(self, vehicle_id, position, length, direction, is_target=False, color_name='white'):
        self.id = vehicle_id
        self.position = position
        self.length = length
        self.direction = direction
        self.is_target = is_target
        self.color_name = color_name
        self.entity = None

    def get_cells(self):
        cells = []
        if self.direction == HORIZONTAL:
            for i in range(self.length):
                cells.append(self.position + i)
        else:
            for i in range(self.length):
                cells.append(self.position + i * BOARD_COLS)
        return cells

    def clone(self):
        return Vehicle(self.id, self.position, self.length, self.direction, self.is_target, self.color_name)

class Board:
    def __init__(self, vehicles):
        self.vehicles = vehicles
        self.occupied = set()
        self.update_occupied()

    def update_occupied(self):
        self.occupied.clear()
        for v in self.vehicles:
            for c in v.get_cells():
                self.occupied.add(c)

    def get_state(self):
        return tuple(v.position for v in self.vehicles)

    def check_win(self):
        for v in self.vehicles:
            if v.is_target:
                cells = v.get_cells()
                if v.direction == HORIZONTAL and cells[-1] % BOARD_COLS == TARGET_EXIT_COL and cells[-1] // BOARD_COLS == TARGET_EXIT_ROW:
                    return True
        return False

    def get_possible_moves(self):
        moves = []
        for idx, v in enumerate(self.vehicles):
            cells = v.get_cells()
            if v.direction == HORIZONTAL:
                row = cells[0] // BOARD_COLS
                # left
                for c in range(cells[0] - 1, row * BOARD_COLS - 1, -1):
                    if c in self.occupied:
                        break
                    moves.append((idx, c))
                # right
                for c in range(cells[-1] + 1, (row + 1) * BOARD_COLS):
                    if c in self.occupied:
                        break
                    moves.append((idx, c))
            else:
                # up
                for c in range(cells[0] - BOARD_COLS, -1, -BOARD_COLS):
                    if c in self.occupied:
                        break
                    moves.append((idx, c))
                # down
                for c in range(cells[-1] + BOARD_COLS, BOARD_COLS * BOARD_ROWS, BOARD_COLS):
                    if c in self.occupied:
                        break
                    moves.append((idx, c))
        return moves

    def apply_move(self, v_idx, new_pos):
        new_vehicles = [v.clone() for v in self.vehicles]
        new_vehicles[v_idx].position = new_pos
        return Board(new_vehicles)

    def can_move_to(self, v_idx, target):
        v = self.vehicles[v_idx]
        cells = v.get_cells()
        if v.direction == HORIZONTAL:
            if target // BOARD_COLS != cells[0] // BOARD_COLS:
                return False
            step = 1 if target > cells[-1] else -1
            for c in range(cells[-1] + step, target + step, step):
                if c in self.occupied:
                    return False
        else:
            if target % BOARD_COLS != cells[0] % BOARD_COLS:
                return False
            step = BOARD_COLS if target > cells[-1] else -BOARD_COLS
            for c in range(cells[-1] + step, target + step, step):
                if c in self.occupied:
                    return False
        return True


def solve_bfs(initial_board):
    start_state = initial_board.get_state()
    queue = collections.deque([(initial_board, [])])
    visited = {start_state}
    while queue:
        board, path = queue.popleft()
        if board.check_win():
            return path
        for v_idx, new_pos in board.get_possible_moves():
            new_board = board.apply_move(v_idx, new_pos)
            new_state = new_board.get_state()
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_board, path + [(v_idx, new_pos)]))
    return None


class RushHourUrsina:
    def __init__(self):
        self.app = Ursina(borderless=False)
        window.title = 'Rush Hour 3D - Core'
        window.size = (1024, 768)

        self.board = None
        self.levels = []
        self.current_level_idx = 0
        self.selected_vehicle_idx = None
        self.move_history = []
        self.moves_count = 0
        self.start_time = 0.0
        self._frozen = False
        self._active_hint_move = None
        self._grid_entities = []
        self._vehicle_entities = []
        self._vehicle_models = {}
        self._hud_entities = []

        self.optimal_moves_cache = {}

        AmbientLight(color=color.rgba(100, 100, 100, 255))
        DirectionalLight(color=color.rgba(200, 200, 200, 255), direction=Vec3(1, -1, -1))

        self._setup_camera()
        self.init_levels()
        self._build_scene()
        self.load_level(0)

    # ------------------ Camera ------------------
    def _setup_camera(self):
        camera.position = Vec3(3.5, 6.5, 7)
        camera.look_at(Vec3(2.5, 0, 2.5))

    # ------------------ Helpers ------------------
    def _cell_to_world(self, cell):
        x = cell % BOARD_COLS
        z = cell // BOARD_COLS
        return Vec3(x, 0.25, z)

    def _cell_center(self, cell):
        return self._cell_to_world(cell)

    def _destroy_all(self, entities):
        for e in entities:
            if e:
                destroy(e)
        entities.clear()

    # ------------------ Scene Builder ------------------
    def _build_scene(self):
        # Ground
        Entity(model='quad', scale=Vec3(6.5, 6.5, 1), rotation_x=90,
               position=Vec3(2.5, 0, 2.5), color=color.dark_gray,
               shader=unlit_shader)

        # Grid cells
        self._destroy_all(self._grid_entities)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cell = r * BOARD_COLS + c
                is_exit = (r == TARGET_EXIT_ROW and c == TARGET_EXIT_COL)
                clr = color.green if is_exit else color.gray
                e = Entity(model='cube', scale=Vec3(0.94, 0.05, 0.94),
                           position=Vec3(c, 0.02, r),
                           color=clr, shader=unlit_shader)
                e.cell_idx = cell
                self._grid_entities.append(e)

        # HUD
        self._destroy_all(self._hud_entities)
        y_base = 0.35
        self.level_text   = Text(text='Level: 1', position=(-0.85, y_base), scale=1.5, color=color.white)
        self.moves_text   = Text(text='Moves: 0', position=(-0.85, y_base - 0.06), scale=1.2, color=color.white)
        self.status_text  = Text(text='',        position=(-0.85, y_base - 0.12), scale=1.2, color=color.yellow)

        self._hud_entities.extend([self.level_text, self.moves_text, self.status_text])

        # Side buttons
        self._make_btn('Undo (Z)',   -0.85, y_base - 0.20, self.undo_move)
        self._make_btn('Reset (R)',  -0.85, y_base - 0.27, self.reset_level)
        self._make_btn('Hint (H)',   -0.85, y_base - 0.34, self.show_hint)
        self._make_btn('Next Level', -0.85, y_base - 0.41, self._next_level)
        self._make_btn('Prev Level', -0.85, y_base - 0.48, self._prev_level)

    def _make_btn(self, label, x, y, callback):
        btn = Button(text=label, position=(x, y), scale=(0.25, 0.04),
                     color=color.azure, text_color=color.white,
                     highlight_color=color.cyan)
        btn.on_click = callback
        self._hud_entities.append(btn)
        return btn

    # ------------------ Level Init ------------------
    def init_levels(self):
        """4 levels with increasing difficulty"""
        b1 = [
            Vehicle('t', 16, 3, HORIZONTAL, False, 'purple'),
            Vehicle('a', 13, 2, VERTICAL,   False, 'cyan'),
            Vehicle('x', 12, 2, HORIZONTAL, True,  'red'),
            Vehicle('b', 15, 2, VERTICAL,   False, 'yellow'),
            Vehicle('c', 21, 2, HORIZONTAL, False, 'blue'),
            Vehicle('d', 27, 2, HORIZONTAL, False, 'green'),
        ]
        b2 = [
            Vehicle('t', 14, 3, VERTICAL,   False, 'purple'),
            Vehicle('a', 19, 2, VERTICAL,   False, 'cyan'),
            Vehicle('x', 12, 2, HORIZONTAL, True,  'red'),
            Vehicle('b', 25, 2, HORIZONTAL, False, 'yellow'),
            Vehicle('c', 22, 2, VERTICAL,   False, 'blue'),
            Vehicle('d', 28, 2, HORIZONTAL, False, 'green'),
            Vehicle('e', 32, 2, HORIZONTAL, False, 'orange'),
        ]
        b3 = [
            Vehicle('t1', 4,  3, VERTICAL,   False, 'purple'),
            Vehicle('t2', 30, 3, HORIZONTAL, False, 'brown'),
            Vehicle('a',  14, 2, VERTICAL,   False, 'cyan'),
            Vehicle('x',  12, 2, HORIZONTAL, True,  'red'),
            Vehicle('b',  19, 2, HORIZONTAL, False, 'yellow'),
            Vehicle('c',  28, 2, VERTICAL,   False, 'blue'),
        ]
        b4 = [
            Vehicle('t1', 3,  3, VERTICAL,   False, 'purple'),
            Vehicle('t2', 20, 3, HORIZONTAL, False, 'brown'),
            Vehicle('a',  0,  2, VERTICAL,   False, 'cyan'),
            Vehicle('b',  7,  2, VERTICAL,   False, 'yellow'),
            Vehicle('x',  12, 2, HORIZONTAL, True,  'red'),
            Vehicle('c',  16, 2, HORIZONTAL, False, 'blue'),
            Vehicle('d',  21, 2, HORIZONTAL, False, 'green'),
            Vehicle('e',  26, 2, VERTICAL,   False, 'orange'),
            Vehicle('f',  29, 2, VERTICAL,   False, 'pink'),
            Vehicle('g',  32, 2, HORIZONTAL, False, 'lime'),
            Vehicle('h',  34, 2, HORIZONTAL, False, 'olive'),
        ]
        self.levels = [Board(b1), Board(b2), Board(b3), Board(b4)]

    def _get_optimal_moves(self, idx):
        if idx not in self.optimal_moves_cache:
            path = solve_bfs(self.levels[idx])
            self.optimal_moves_cache[idx] = len(path) if path else 999
        return self.optimal_moves_cache[idx]

    def _calc_stars(self, moves, optimal, elapsed):
        diff = moves - int(optimal)
        if diff <= 1 and elapsed <= 45.0:
            return 3
        if diff <= 4 and elapsed <= 90.0:
            return 2
        return 1

    # ------------------ Load / Reset ------------------
    def load_level(self, idx):
        if idx < 0 or idx >= len(self.levels):
            return
        self.current_level_idx = idx
        orig = self.levels[idx]
        self.board = Board([v.clone() for v in orig.vehicles])
        self.move_history = []
        self.moves_count = 0
        self.start_time = time.time()
        self._frozen = False
        self.selected_vehicle_idx = None
        self._active_hint_move = None

        self._rebuild_vehicles()
        self._update_hud()
        self.status_text.text = ''

    def reset_level(self):
        self.load_level(self.current_level_idx)

    def _next_level(self):
        self.load_level(self.current_level_idx + 1)

    def _prev_level(self):
        self.load_level(self.current_level_idx - 1)

    # ------------------ Vehicle Rendering ------------------
    def _rebuild_vehicles(self):
        self._destroy_all(self._vehicle_entities)
        for i, v in enumerate(self.board.vehicles):
            cells = v.get_cells()
            if v.direction == HORIZONTAL:
                w, d = v.length, 1
                cx = cells[0] % BOARD_COLS + (v.length - 1) / 2.0
                cz = cells[0] // BOARD_COLS
            else:
                w, d = 1, v.length
                cx = cells[0] % BOARD_COLS
                cz = cells[0] // BOARD_COLS + (v.length - 1) / 2.0

            if v.is_target:
                clr = color.red
            elif v.color_name == 'purple':
                clr = color.magenta
            elif v.color_name == 'cyan':
                clr = color.cyan
            elif v.color_name == 'yellow':
                clr = color.yellow
            elif v.color_name == 'blue':
                clr = color.blue
            elif v.color_name == 'green':
                clr = color.green
            elif v.color_name == 'orange':
                clr = color.orange
            elif v.color_name == 'brown':
                clr = color.brown
            elif v.color_name == 'pink':
                clr = color.pink
            elif v.color_name == 'lime':
                clr = color.lime
            elif v.color_name == 'olive':
                clr = color.olive
            else:
                clr = color.white

            e = Entity(model='cube', scale=Vec3(w * 0.9, 0.35, d * 0.9),
                       position=Vec3(cx, 0.25, cz),
                       color=clr, shader=basic_lighting_shader,
                       collider='box')
            e.vehicle_idx = i
            self._vehicle_entities.append(e)

    def _sync_vehicle_entities(self):
        for i, v in enumerate(self.board.vehicles):
            if i < len(self._vehicle_entities):
                e = self._vehicle_entities[i]
                cells = v.get_cells()
                if v.direction == HORIZONTAL:
                    cx = cells[0] % BOARD_COLS + (v.length - 1) / 2.0
                    cz = cells[0] // BOARD_COLS
                else:
                    cx = cells[0] % BOARD_COLS
                    cz = cells[0] // BOARD_COLS + (v.length - 1) / 2.0
                e.position = Vec3(cx, 0.25, cz)

    # ------------------ HUD ------------------
    def _update_hud(self):
        self.level_text.text = f'Level: {self.current_level_idx + 1}'
        self.moves_text.text = f'Moves: {self.moves_count}'

    # ------------------ Undo ------------------
    def undo_move(self):
        if self._frozen or not self.move_history:
            return
        v_idx, old_pos = self.move_history.pop()
        self.board.vehicles[v_idx].position = old_pos
        self.board.update_occupied()
        self.moves_count = max(0, self.moves_count - 1)
        self.selected_vehicle_idx = None
        self._active_hint_move = None
        self._sync_vehicle_entities()
        self._update_hud()

    # ------------------ Hint ------------------
    def show_hint(self):
        if self._frozen:
            return
        path = solve_bfs(self.board)
        if path:
            v_idx, new_pos = path[0]
            self._active_hint_move = (v_idx, new_pos)
            self.status_text.text = f"HINT: Move {self.board.vehicles[v_idx].id}"
            self.status_text.color = color.yellow
        else:
            self.status_text.text = "No solution found!"
            self.status_text.color = color.red

    # ------------------ Move Execution ------------------
    def _try_move_to_cell(self, v_idx, target_cell):
        if self._frozen:
            return False
        v = self.board.vehicles[v_idx]
        cells = v.get_cells()
        if v.direction == HORIZONTAL:
            if target_cell // BOARD_COLS != cells[0] // BOARD_COLS:
                return False
            best = cells[0]
            for p in cells:
                if abs(p - target_cell) < abs(best - target_cell):
                    best = p
            new_pos = best - (cells[0] - v.position)
            if not self.board.can_move_to(v_idx, new_pos + v.length - 1 if new_pos > v.position else new_pos):
                return False
        else:
            if target_cell % BOARD_COLS != cells[0] % BOARD_COLS:
                return False
            best = cells[0]
            for p in cells:
                if abs(p - target_cell) < abs(best - target_cell):
                    best = p
            new_pos = best - (cells[0] - v.position)
            if not self.board.can_move_to(v_idx, new_pos + (v.length - 1) * BOARD_COLS if new_pos > v.position else new_pos):
                return False
        self._execute_move(v_idx, new_pos)
        return True

    def _execute_move(self, v_idx, new_pos):
        old_pos = self.board.vehicles[v_idx].position
        self.move_history.append((v_idx, old_pos))
        self.board.vehicles[v_idx].position = new_pos
        self.board.update_occupied()
        self.moves_count += 1

        e = self._vehicle_entities[v_idx]
        cells = self.board.vehicles[v_idx].get_cells()
        v = self.board.vehicles[v_idx]
        if v.direction == HORIZONTAL:
            cx = cells[0] % BOARD_COLS + (v.length - 1) / 2.0
            cz = cells[0] // BOARD_COLS
        else:
            cx = cells[0] % BOARD_COLS
            cz = cells[0] // BOARD_COLS + (v.length - 1) / 2.0
        e.animate_position(Vec3(cx, 0.25, cz), duration=0.25, curve=curve.out_quad)

        self._active_hint_move = None
        self._update_hud()

        if self.board.check_win():
            self._on_victory()

    def _on_victory(self):
        self._frozen = True
        elapsed = time.time() - self.start_time
        optimal = self._get_optimal_moves(self.current_level_idx)
        stars = self._calc_stars(self.moves_count, optimal, elapsed)
        star_str = 'â˜?' * stars + 'â˜?' * (3 - stars)
        self.status_text.text = f"WIN! {star_str}  {self.moves_count} moves  {elapsed:.1f}s"
        self.status_text.color = color.gold
        invoke(self._next_level, delay=2.5)

    # ------------------ Input ------------------
    def _input(self, key):
        if self._frozen:
            return
        if key == 'z':
            self.undo_move()
        elif key == 'r':
            self.reset_level()
        elif key == 'h':
            self.show_hint()

    def _update(self):
        if self._frozen:
            return
        if mouse.left:
            if mouse.hovered_entity:
                ent = mouse.hovered_entity
                # Click vehicle to select
                if hasattr(ent, 'vehicle_idx'):
                    self.selected_vehicle_idx = ent.vehicle_idx
                    self.status_text.text = f"Selected: {self.board.vehicles[ent.vehicle_idx].id}"
                    self.status_text.color = color.white
                # Click grid cell to move selected vehicle
                elif hasattr(ent, 'cell_idx') and self.selected_vehicle_idx is not None:
                    self._try_move_to_cell(self.selected_vehicle_idx, ent.cell_idx)


_game = None

def input(key):
    if _game and hasattr(_game, '_input'):
        try:
            _game._input(key)
        except Exception:
            traceback.print_exc()

def update():
    if _game and hasattr(_game, '_update'):
        try:
            _game._update()
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    _game = RushHourUrsina()
    _game.app.run()
