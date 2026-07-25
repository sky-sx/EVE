"""EVE 控制与监视面板 — 朴素但功能完整的 GUI。"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, scrolledtext
import time
import io
from typing import Any

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class ControlPanel:
    """EVE 主控制窗口。

    显示运行时状态，提供控制按钮。
    使用 tkinter（Python 内置，无需额外依赖）。
    """

    def __init__(self, on_cold_start=None, on_emergency_stop=None,
                 on_praise=None, on_criticize=None, on_force_sleep=None,
                 on_manual_save=None):
        self._root = tk.Tk()
        self._root.title("EVE Control Panel")
        self._root.geometry("1200x800")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Callbacks
        self._on_cold_start = on_cold_start
        self._on_emergency_stop = on_emergency_stop
        self._on_praise = on_praise
        self._on_criticize = on_criticize
        self._on_force_sleep = on_force_sleep
        self._on_manual_save = on_manual_save

        # State reference (set externally by bind_state)
        self._state = None       # RuntimeState
        self._world = None       # WorldState
        self._myself = None      # MyselfState
        self._blackboard = None  # Blackboard
        self._hormones = None    # HormoneManager
        self._graph = None       # TNNGraph
        self._tnn_store = None   # TNNStore
        self._memorizer = None   # Memorizer
        self._buffer = None      # InputBuffer
        self._trainer = None     # Trainer
        self._config = None      # EVEConfig

        self._running = False
        self._log_lines: list[str] = []
        self._max_log_lines = 500
        self._last_cursor_x: int = 0
        self._last_cursor_y: int = 0
        self._photo: Any = None  # Screen photo reference
        self._build_ui()

    def bind_state(self, state, world, myself, blackboard, hormones,
                   graph, tnn_store, memorizer, buffer, trainer, config):
        """绑定运行时状态引用。GUI 通过 after() 定时刷新。"""
        self._state = state
        self._world = world
        self._myself = myself
        self._blackboard = blackboard
        self._hormones = hormones
        self._graph = graph
        self._tnn_store = tnn_store
        self._memorizer = memorizer
        self._buffer = buffer
        self._trainer = trainer
        self._config = config

    def _build_ui(self):
        """构建界面。"""
        # Main paned window
        paned = ttk.PanedWindow(self._root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── Left panel: Controls ──
        left = ttk.Frame(paned, width=320)
        paned.add(left, weight=0)
        self._build_left_panel(left)

        # ── Right panel: Tabs ──
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        """左侧控制面板。"""
        # Canvas + Scrollbar for scrollable left panel
        canvas = tk.Canvas(parent, width=310, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Make mousewheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Screen preview ──
        screen_frame = ttk.LabelFrame(scroll_frame, text="Screen", padding=3)
        screen_frame.pack(fill=tk.X, padx=3, pady=2)
        self._screen_label = ttk.Label(screen_frame, text="(no frame)", anchor=tk.CENTER, background="#1a1a1a", foreground="#888")
        self._screen_label.pack(fill=tk.X, pady=2)

        # ── Cursor position ──
        cursor_frame = ttk.LabelFrame(scroll_frame, text="Cursor", padding=3)
        cursor_frame.pack(fill=tk.X, padx=3, pady=2)
        self._cursor_label = ttk.Label(cursor_frame, text="x=--, y=--", font=("Consolas", 10))
        self._cursor_label.pack(anchor=tk.W)

        # ── Control buttons ──
        ctrl_frame = ttk.LabelFrame(scroll_frame, text="Controls", padding=3)
        ctrl_frame.pack(fill=tk.X, padx=3, pady=2)

        self._btn_cold_start = ttk.Button(ctrl_frame, text="❄ Cold Start", command=self._do_cold_start)
        self._btn_cold_start.pack(fill=tk.X, pady=1)

        self._btn_emergency = ttk.Button(ctrl_frame, text="⚠ EMERGENCY STOP",
                                          command=self._do_emergency_stop)
        self._btn_emergency.pack(fill=tk.X, pady=1)

        # ── Output mode & permissions ──
        out_frame = ttk.LabelFrame(scroll_frame, text="Output Mode & Permissions", padding=3)
        out_frame.pack(fill=tk.X, padx=3, pady=2)

        self._mode_label = ttk.Label(out_frame, text="Mode: disabled", font=("", 9, "bold"))
        self._mode_label.pack(anchor=tk.W)

        self._mouse_var = tk.BooleanVar(value=False)
        self._mouse_cb = ttk.Checkbutton(out_frame, text="Mouse", variable=self._mouse_var, command=self._on_mouse_toggle)
        self._mouse_cb.pack(anchor=tk.W)

        self._keyboard_var = tk.BooleanVar(value=False)
        self._keyboard_cb = ttk.Checkbutton(out_frame, text="Keyboard", variable=self._keyboard_var, command=self._on_keyboard_toggle)
        self._keyboard_cb.pack(anchor=tk.W)

        self._speak_var = tk.BooleanVar(value=False)
        self._speak_cb = ttk.Checkbutton(out_frame, text="Speak", variable=self._speak_var, command=self._on_speak_toggle)
        self._speak_cb.pack(anchor=tk.W)

        # Mode radio buttons
        mode_frame = ttk.Frame(out_frame)
        mode_frame.pack(fill=tk.X, pady=2)
        self._mode_var = tk.StringVar(value="disabled")
        for mode in ["disabled", "mock", "real"]:
            ttk.Radiobutton(mode_frame, text=mode, variable=self._mode_var,
                            value=mode, command=self._on_mode_change).pack(side=tk.LEFT)

        # ── Safegate status ──
        sg_frame = ttk.LabelFrame(scroll_frame, text="Safegate", padding=3)
        sg_frame.pack(fill=tk.X, padx=3, pady=2)
        self._safegate_label = ttk.Label(sg_frame, text="OK", foreground="green")
        self._safegate_label.pack(anchor=tk.W)

        # ── Interaction buttons ──
        int_frame = ttk.LabelFrame(scroll_frame, text="Interaction", padding=3)
        int_frame.pack(fill=tk.X, padx=3, pady=2)

        ttk.Button(int_frame, text="👍 Praise", command=self._do_praise).pack(fill=tk.X, pady=1)
        ttk.Button(int_frame, text="👎 Criticize", command=self._do_criticize).pack(fill=tk.X, pady=1)
        ttk.Button(int_frame, text="💤 Force Sleep", command=self._do_force_sleep).pack(fill=tk.X, pady=1)
        ttk.Button(int_frame, text="💾 Save Snapshot", command=self._do_manual_save).pack(fill=tk.X, pady=1)

        # ── Loop status ──
        loop_frame = ttk.LabelFrame(scroll_frame, text="Loop Status", padding=3)
        loop_frame.pack(fill=tk.X, padx=3, pady=2)
        self._loop_label = ttk.Label(loop_frame, text="Not started", foreground="gray")
        self._loop_label.pack(anchor=tk.W)

        # ── Dock status ──
        dock_frame = ttk.LabelFrame(scroll_frame, text="Dock", padding=3)
        dock_frame.pack(fill=tk.X, padx=3, pady=2)
        self._dock_status_label = ttk.Label(dock_frame, text="Idle")
        self._dock_status_label.pack(anchor=tk.W)

        # ── Status bar ──
        status_frame = ttk.Frame(scroll_frame)
        status_frame.pack(fill=tk.X, padx=3, pady=2)
        self._status_label = ttk.Label(status_frame, text="Ready", foreground="gray")
        self._status_label.pack(anchor=tk.W)

    def _build_right_panel(self, parent):
        """右侧分页显示。"""
        self._notebook = ttk.Notebook(parent)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Overview
        tab1 = ttk.Frame(self._notebook)
        self._notebook.add(tab1, text="Overview")
        self._build_overview_tab(tab1)

        # Tab 2: Memory
        tab2 = ttk.Frame(self._notebook)
        self._notebook.add(tab2, text="Memory")
        self._build_memory_tab(tab2)

        # Tab 3: TNN
        tab3 = ttk.Frame(self._notebook)
        self._notebook.add(tab3, text="TNN")
        self._build_tnn_tab(tab3)

        # Tab 4: Hormones
        tab4 = ttk.Frame(self._notebook)
        self._notebook.add(tab4, text="Hormones")
        self._build_hormone_tab(tab4)

        # Tab 5: Logs
        tab5 = ttk.Frame(self._notebook)
        self._notebook.add(tab5, text="Logs")
        self._build_log_tab(tab5)

    def _build_overview_tab(self, parent):
        """Overview tab: world summary, myself summary, thinking, blackboard."""
        pw = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        pw.pack(fill=tk.BOTH, expand=True)

        # World
        wf = ttk.LabelFrame(pw, text="World", padding=3)
        pw.add(wf, weight=1)
        self._world_text = tk.Text(wf, height=5, width=80, wrap=tk.WORD)
        self._world_text.pack(fill=tk.BOTH, expand=True, pady=2)

        # Myself
        mf = ttk.LabelFrame(pw, text="Myself", padding=3)
        pw.add(mf, weight=1)
        self._myself_text = tk.Text(mf, height=5, width=80, wrap=tk.WORD)
        self._myself_text.pack(fill=tk.BOTH, expand=True, pady=2)

        # Thinking
        tf = ttk.LabelFrame(pw, text="What I'm Thinking", padding=3)
        pw.add(tf, weight=1)
        self._thinking_text = tk.Text(tf, height=3, width=80, wrap=tk.WORD, foreground="#0055aa")
        self._thinking_text.pack(fill=tk.BOTH, expand=True, pady=2)

        # Blackboard
        bf = ttk.LabelFrame(pw, text="Blackboard", padding=3)
        pw.add(bf, weight=1)
        self._blackboard_text = tk.Text(bf, height=4, width=80, wrap=tk.WORD)
        self._blackboard_text.pack(fill=tk.BOTH, expand=True, pady=2)

    def _build_memory_tab(self, parent):
        """Memory stats display."""
        self._mem_text = tk.Text(parent, width=80, wrap=tk.WORD)
        self._mem_text.pack(fill=tk.BOTH, expand=True, pady=2)

        # Configure tags for coloring
        self._mem_text.tag_config("stm", foreground="#228833")
        self._mem_text.tag_config("mtm", foreground="#ee7733")
        self._mem_text.tag_config("ltm", foreground="#0077bb")
        self._mem_text.tag_config("header", font=("", 10, "bold"))

    def _build_tnn_tab(self, parent):
        """TNN list display."""
        pw = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        pw.pack(fill=tk.BOTH, expand=True)

        # Available TNNs
        af = ttk.LabelFrame(pw, text="Available TNNs", padding=3)
        pw.add(af, weight=1)
        self._available_tnn_text = tk.Text(af, height=6, width=80, wrap=tk.WORD,
                                            font=("Consolas", 9))
        self._available_tnn_text.pack(fill=tk.BOTH, expand=True, pady=2)

        # Loaded TNNs
        lf = ttk.LabelFrame(pw, text="Loaded TNNs (in Graph)", padding=3)
        pw.add(lf, weight=1)
        self._loaded_tnn_text = tk.Text(lf, height=6, width=80, wrap=tk.WORD,
                                         font=("Consolas", 9))
        self._loaded_tnn_text.pack(fill=tk.BOTH, expand=True, pady=2)

        # Dock status
        df = ttk.LabelFrame(pw, text="Dock Status", padding=3)
        pw.add(df, weight=1)
        self._dock_text = tk.Text(df, height=5, width=80, wrap=tk.WORD,
                                   font=("Consolas", 9))
        self._dock_text.pack(fill=tk.BOTH, expand=True, pady=2)

    def _build_hormone_tab(self, parent):
        """Hormone levels display with progress bars."""
        self._hormone_bars: dict[str, ttk.Progressbar] = {}
        self._hormone_labels: dict[str, ttk.Label] = {}
        self._tendency_labels: dict[str, ttk.Label] = {}

        hormones = [
            ("dopamine", "Dopamine (explore/seeking)"),
            ("serotonin", "Serotonin (well-being)"),
            ("norepinephrine", "Norepinephrine (alertness)"),
            ("oxytocin", "Oxytocin (social bond)"),
            ("cortisol", "Cortisol (stress)"),
            ("acetylcholine", "Acetylcholine (learning)"),
        ]

        for i, (key, desc) in enumerate(hormones):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, padx=10, pady=3)

            label = ttk.Label(row, text=f"{desc}:", width=30)
            label.pack(side=tk.LEFT)

            bar = ttk.Progressbar(row, length=200, mode="determinate", maximum=100)
            bar.pack(side=tk.LEFT, padx=5)

            val_label = ttk.Label(row, text="0.50", width=6, font=("Consolas", 9))
            val_label.pack(side=tk.LEFT)

            self._hormone_bars[key] = bar
            self._hormone_labels[key] = val_label

        # Tendencies
        tend_frame = ttk.LabelFrame(parent, text="Tendencies", padding=5)
        tend_frame.pack(fill=tk.X, padx=10, pady=10)

        tendency_names = [
            ("explore", "Explore"),
            ("exploit", "Exploit"),
            ("pause", "Pause"),
            ("sleep", "Sleep urge"),
            ("active_output", "Active output"),
            ("think_more", "Think more"),
            ("train", "Train"),
        ]

        for i, (key, desc) in enumerate(tendency_names):
            row = ttk.Frame(tend_frame)
            row.pack(fill=tk.X, padx=5, pady=1)

            label = ttk.Label(row, text=f"{desc}:", width=18)
            label.pack(side=tk.LEFT)

            bar = ttk.Progressbar(row, length=150, mode="determinate", maximum=100)
            bar.pack(side=tk.LEFT, padx=5)

            val_label = ttk.Label(row, text="0.000", width=8, font=("Consolas", 9))
            val_label.pack(side=tk.LEFT)

            self._tendency_labels[key] = val_label

    def _build_log_tab(self, parent):
        """Log display."""
        self._log_text = scrolledtext.ScrolledText(parent, width=80, font=("Consolas", 9), wrap=tk.WORD)
        self._log_text.pack(fill=tk.BOTH, expand=True, pady=2)
        self._log_text.config(state=tk.DISABLED)

    def start(self) -> None:
        """启动 GUI 主循环。"""
        self._running = True
        self._root.bind("<Escape>", lambda e: self._do_emergency_stop())
        self._refresh()
        self._root.mainloop()

    def stop(self) -> None:
        """停止 GUI。"""
        self._running = False
        try:
            self._root.quit()
        except Exception:
            pass

    def _refresh(self):
        """定时刷新界面（每 200ms）。"""
        if not self._running:
            return
        try:
            self._update_display()
        except Exception:
            pass
        self._root.after(200, self._refresh)

    def _update_display(self):
        """根据当前状态更新所有显示。"""
        if self._state is None:
            return

        # Update controls from state
        self._update_controls_state()

        # Update screen preview
        self._update_screen()

        # Update cursor
        self._update_cursor()

        # Update Overview tab
        self._update_overview()

        # Update Memory tab
        self._update_memory()

        # Update TNN tab
        self._update_tnn()

        # Update Hormone tab
        self._update_hormones()

        # Update Dock
        self._update_dock()

        # Update Loop status
        self._update_loop()

        # Update status
        self._update_status()

    def _update_controls_state(self):
        """Sync control widgets with current RuntimeState."""
        s = self._state
        if self._mode_var.get() != s.output_mode.value:
            self._mode_var.set(s.output_mode.value)
        if self._mouse_var.get() != s.mouse_allowed:
            self._mouse_var.set(s.mouse_allowed)
        if self._keyboard_var.get() != s.keyboard_allowed:
            self._keyboard_var.set(s.keyboard_allowed)
        if self._speak_var.get() != s.speak_allowed:
            self._speak_var.set(s.speak_allowed)

        # Safegate status
        if s.emergency_stopped:
            self._safegate_label.config(text="EMERGENCY STOPPED", foreground="red")
        elif not s.cold_started:
            self._safegate_label.config(text="Not started", foreground="orange")
        elif s.output_mode.value == "disabled":
            self._safegate_label.config(text="Disabled", foreground="orange")
        elif s.blocked_until_ns > time.monotonic_ns():
            remaining = (s.blocked_until_ns - time.monotonic_ns()) / 1e9
            self._safegate_label.config(text=f"Frozen {remaining:.1f}s", foreground="orange")
        else:
            self._safegate_label.config(text="OK", foreground="green")

        # Button states
        if s.cold_started:
            self._btn_cold_start.config(state=tk.DISABLED)
        else:
            self._btn_cold_start.config(state=tk.NORMAL)

    def _update_screen(self):
        """Display latest screen frame from buffer."""
        if self._buffer is None:
            return
        sample = self._buffer.latest("screen")
        if sample is None:
            self._screen_label.config(image="", text="(no frame)")
            return

        frame = sample.value
        if not _PIL_AVAILABLE:
            self._screen_label.config(
                image="",
                text=f"({frame.shape[1]}x{frame.shape[0]}, PIL not installed)"
            )
            return

        try:
            # Resize for preview
            h, w = frame.shape[:2]
            scale = min(280 / w, 160 / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)

            if new_w > 0 and new_h > 0:
                img = Image.fromarray(frame)
                img = img.resize((new_w, new_h), Image.NEAREST)
                self._photo = ImageTk.PhotoImage(img)
                self._screen_label.config(image=self._photo, text="")
            else:
                self._screen_label.config(image="", text="(resize failed)")
        except Exception:
            self._screen_label.config(image="", text="(frame render error)")

    def _update_cursor(self):
        """Show latest cursor position."""
        if self._buffer is None:
            return
        sample = self._buffer.latest("cursor")
        if sample is not None:
            x, y = sample.value
            self._last_cursor_x = int(x)
            self._last_cursor_y = int(y)
            self._cursor_label.config(text=f"x={int(x)}, y={int(y)}")
        else:
            self._cursor_label.config(text=f"x={self._last_cursor_x}, y={self._last_cursor_y}")

    def _update_overview(self):
        """Update world, myself, thinking, blackboard displays."""
        if self._world is not None:
            world_lines = [
                f"Scene: {self._world.scene or '(none)'}",
                f"Sub-scene: {self._world.sub_scene or '(none)'}",
                f"Window: {self._world.active_window or '(none)'}",
                f"Objects: {', '.join(self._world.visible_objects) if self._world.visible_objects else '(none)'}",
                f"Text: {self._world.detected_text[:120] if self._world.detected_text else '(none)'}",
                f"Uncertainty: {self._world.uncertainty or '(none)'}",
            ]
            _set_text(self._world_text, "\n".join(world_lines))

        if self._myself is not None:
            myself_lines = [
                f"Current task: {self._myself.current_task or '(none)'}",
                f"Task progress: {self._myself.task_progress or '(none)'}",
                f"Loaded TNNs: {', '.join(self._myself.loaded_tnn) if self._myself.loaded_tnn else '(none)'}",
                f"Available: {', '.join(self._myself.available_tnn_summary) if self._myself.available_tnn_summary else '(none)'}",
            ]
            if self._myself.resource_status:
                myself_lines.append(f"Resources: {self._myself.resource_status}")
            _set_text(self._myself_text, "\n".join(myself_lines))

            thinking = self._myself.what_im_thinking or "(not thinking yet)"
            _set_text(self._thinking_text, thinking)

        if self._blackboard is not None:
            bb_lines: list[str] = []
            for kind, entries in self._blackboard.entries.items():
                bb_lines.append(f"[{kind}] {len(entries)} entries")
                for e in entries[-3:]:
                    bb_lines.append(f"  - {e.entry_id}: {str(e.payload)[:80]}")
            if not bb_lines:
                bb_lines.append("(empty)")
            _set_text(self._blackboard_text, "\n".join(bb_lines))

    def _update_memory(self):
        """Update STM/MTM/LTM stats."""
        if self._memorizer is None:
            return

        stm_ids = self._memorizer.get_stm_ids()
        mtm_ids = self._memorizer.get_mtm_ids()

        lines = [
            ("header", "=== Memory Statistics ==="),
            ("", ""),
            ("stm", f"STM:  {len(stm_ids)} entries"),
            ("mtm", f"MTM:  {len(mtm_ids)} entries"),
        ]

        try:
            stats = self._memorizer.stats()
            lines.append(("ltm", f"LTM:  {stats.get('ltm_count', 0)} entries"))
            lines.append(("", f"Total size: {stats.get('total_size_bytes', 0):,} bytes"))
        except Exception:
            lines.append(("ltm", "LTM:  (stats unavailable)"))

        _set_text_with_tags(self._mem_text, lines)

    def _update_tnn(self):
        """Update TNN lists and Dock status."""
        # Available TNNs
        if self._tnn_store is not None:
            available = self._tnn_store.list_available()
            if available:
                avail_lines = []
                for tid in available:
                    desc = self._tnn_store.get_descriptor(tid)
                    if desc:
                        avail_lines.append(f"  {tid}  v{desc.version}  {desc.purpose or ''}")
                    else:
                        avail_lines.append(f"  {tid}")
                _set_text(self._available_tnn_text, "\n".join(avail_lines))
            else:
                _set_text(self._available_tnn_text, "(no TNNs available)")
        else:
            _set_text(self._available_tnn_text, "(tnn_store not bound)")

        # Loaded TNNs (in graph)
        if self._graph is not None:
            loaded = self._graph.list_nodes()
            if loaded:
                load_lines: list[str] = []
                for tnn_id in loaded:
                    node = self._graph.get_node(tnn_id)
                    if node:
                        freq = node.run_frequency_hz
                        status = node.status
                        count = node.run_count
                        err = f" ERR:{node.error_message}" if node.error_message else ""
                        load_lines.append(
                            f"  {tnn_id}  freq={freq:.1f}Hz  runs={count}  status={status}{err}"
                        )
                    else:
                        load_lines.append(f"  {tnn_id}")
                _set_text(self._loaded_tnn_text, "\n".join(load_lines))
            else:
                _set_text(self._loaded_tnn_text, "(no TNNs loaded in graph)")
        else:
            _set_text(self._loaded_tnn_text, "(graph not bound)")

    def _update_hormones(self):
        """Update hormone progress bars and tendencies."""
        if self._hormones is None:
            return

        levels = self._hormones.levels
        for key in self._hormone_bars:
            val = getattr(levels, key, 0.5)
            pct = int(val * 100)
            self._hormone_bars[key]["value"] = pct
            self._hormone_labels[key].config(text=f"{val:.2f}")

        tendencies = self._hormones.get_tendencies()
        for key, label in self._tendency_labels.items():
            val = tendencies.get(key, 0.0)
            label.config(text=f"{val:.3f}")

    def _update_dock(self):
        """Update dock status display."""
        if self._trainer is None:
            self._dock_status_label.config(text="(trainer not bound)")
            _set_text(self._dock_text, "(trainer not bound)")
            return

        try:
            stats = self._trainer.stats()
            status_text = "Training" if stats.get("is_training") else "Idle"
            self._dock_status_label.config(
                text=status_text,
                foreground="blue" if stats.get("is_training") else "gray"
            )

            dock_lines = [
                f"Status: {'Training' if stats.get('is_training') else 'Idle'}",
                f"Queue: {stats.get('queue_size', 0)} orders",
                f"Current: {stats.get('current_order') or '(none)'}",
                f"Completed: {stats.get('total_completed', 0)} total "
                f"({stats.get('success_count', 0)} success, {stats.get('fail_count', 0)} failed)",
            ]
            _set_text(self._dock_text, "\n".join(dock_lines))
        except Exception:
            self._dock_status_label.config(text="Idle")
            _set_text(self._dock_text, "(stats unavailable)")

    def _update_loop(self):
        """Update loop status display."""
        if self._state is None:
            return

        parts = []
        if self._state.cold_started:
            parts.append("Running")
            if self._state.emergency_stopped:
                parts.append("EMERGENCY")
            elif self._state.output_mode.value == "disabled":
                parts.append("(disabled)")
        else:
            parts.append("Not started")

        self._loop_label.config(text=" | ".join(parts),
                                 foreground="red" if self._state.emergency_stopped else
                                 "green" if self._state.cold_started else "gray")

    def _update_status(self):
        """Update the status bar text."""
        if self._state is None:
            return

        if self._state.emergency_stopped:
            self._status_label.config(text="EMERGENCY STOPPED", foreground="red")
        elif self._state.cold_started:
            mode = self._state.output_mode.value
            self._status_label.config(text=f"Running ({mode})", foreground="green")
        else:
            self._status_label.config(text="Press 'Cold Start' to begin", foreground="gray")

    def add_log(self, message: str) -> None:
        """添加日志条目。"""
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self._log_lines.append(line)

        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines:]

        # Update widget if it exists
        if hasattr(self, "_log_text"):
            try:
                self._log_text.config(state=tk.NORMAL)
                self._log_text.insert(tk.END, line + "\n")
                self._log_text.see(tk.END)
                self._log_text.config(state=tk.DISABLED)
            except Exception:
                pass

    # ── Button callbacks ──

    def _do_cold_start(self):
        if self._on_cold_start:
            self.add_log("Cold Start triggered")
            self._on_cold_start()

    def _do_emergency_stop(self):
        if self._on_emergency_stop:
            self.add_log("EMERGENCY STOP triggered!")
            self._on_emergency_stop()
        # Also set state directly if callback not provided
        elif self._state is not None:
            self._state.emergency_stopped = True
            self.add_log("EMERGENCY STOP triggered! (direct)")

    def _do_praise(self):
        if self._on_praise:
            self.add_log("User praised EVE")
            self._on_praise()

    def _do_criticize(self):
        if self._on_criticize:
            self.add_log("User criticized EVE")
            self._on_criticize()

    def _do_force_sleep(self):
        if self._on_force_sleep:
            self.add_log("Force sleep triggered")
            self._on_force_sleep()

    def _do_manual_save(self):
        if self._on_manual_save:
            self.add_log("Manual snapshot save requested")
            self._on_manual_save()

    def _on_close(self):
        """窗口关闭处理。"""
        self.add_log("GUI window closing...")
        self._running = False
        try:
            self._root.destroy()
        except Exception:
            pass

    # ── Output control callbacks ──

    def _on_mouse_toggle(self):
        if self._state is not None:
            self._state.mouse_allowed = self._mouse_var.get()

    def _on_keyboard_toggle(self):
        if self._state is not None:
            self._state.keyboard_allowed = self._keyboard_var.get()

    def _on_speak_toggle(self):
        if self._state is not None:
            self._state.speak_allowed = self._speak_var.get()

    def _on_mode_change(self):
        if self._state is not None:
            mode_str = self._mode_var.get()
            from eve.state import OutputMode
            self._state.output_mode = OutputMode(mode_str)
            self.add_log(f"Output mode changed to: {mode_str}")
            self._mode_label.config(text=f"Mode: {mode_str}")


# ── Helpers ───────────────────────────────────────────────

def _set_text(widget: tk.Text, text: str) -> None:
    """Set text widget content, preserving nothing else."""
    current = widget.get("1.0", tk.END).rstrip("\n")
    if current == text:
        return
    widget.config(state=tk.NORMAL)
    widget.delete("1.0", tk.END)
    widget.insert("1.0", text)
    widget.config(state=tk.DISABLED)


def _set_text_with_tags(widget: tk.Text, lines: list[tuple[str, str]]) -> None:
    """Set text widget content with per-line tags."""
    widget.config(state=tk.NORMAL)
    widget.delete("1.0", tk.END)
    for tag, text in lines:
        if tag:
            widget.insert(tk.END, text + "\n", tag)
        else:
            widget.insert(tk.END, text + "\n")
    widget.config(state=tk.DISABLED)
