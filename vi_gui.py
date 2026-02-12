#!/usr/bin/env python3
"""
Value Iteration GUI - Python conversion of VIgui.java

This application demonstrates value iteration for a grid world problem.
Converted from Java Swing to Python tkinter.

Original Java code by David Poole, UBC
Python conversion maintains the same functionality and layout.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
# import numpy as np  # Temporarily disabled due to installation issues
import math
from typing import List, Tuple, Optional


class VICore:
    """
    Core value iteration logic - implements the algorithm referenced by the GUI.
    Based on usage patterns observed in the original Java code.
    """

    def __init__(self):
        self.discount = 0.9
        self.absorbing = False

        # Initialize 10x10 grid
        self.grid_size = 10
        self.values = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Q-values for 4 actions: up, right, down, left
        self.qvalues = [[[0.0 for _ in range(4)] for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Rewards (can be customized)
        self.rewards = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Set up some sample rewards for demonstration
        self.rewards[8][8] = 10.0   # Goal state
        self.rewards[2][2] = -5.0   # Obstacle
        self.rewards[7][3] = -5.0   # Obstacle

    def doreset(self, initial_value: float = 0.0):
        """Reset all values to initial value"""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.values[x][y] = initial_value
                for a in range(4):
                    self.qvalues[x][y][a] = initial_value

    def dostep(self, discount: float):
        """Perform one step of value iteration"""
        self.discount = discount

        # Store old values for convergence check
        old_values = [[self.values[x][y] for y in range(self.grid_size)] for x in range(self.grid_size)]

        # Update Q-values for each state-action pair
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.absorbing and (self.rewards[x][y] != 0):
                    # Absorbing states keep their reward value
                    self.values[x][y] = self.rewards[x][y]
                    continue

                # Calculate Q-values for each action
                actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left

                for action_idx, (dx, dy) in enumerate(actions):
                    next_x, next_y = x + dx, y + dy

                    # Check bounds
                    if 0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size:
                        next_value = old_values[next_x][next_y]
                    else:
                        next_value = old_values[x][y]  # Stay in same state if hitting boundary

                    # Q-value = reward + discount * value of next state
                    self.qvalues[x][y][action_idx] = self.rewards[x][y] + self.discount * next_value

                # Value is max over all Q-values
                self.values[x][y] = max(self.qvalues[x][y])


class GridPanel(tk.Canvas):
    """Custom canvas for drawing the grid visualization"""

    def __init__(self, parent, vi_core: VICore, **kwargs):
        super().__init__(parent, **kwargs)
        self.vi_core = vi_core
        self.sqsize = 50
        self.twid = 5
        self.brightness = 1.0
        self.font_size = 14

        # Configure canvas
        self.configure(width=self.sqsize * 10, height=self.sqsize * 10, bg='white')

    def update_display(self):
        """Redraw the entire grid"""
        self.delete("all")

        # Draw grid cells with values
        for x in range(10):
            for y in range(10):
                value = self.vi_core.values[x][y]

                # Color based on value (green for positive, red for negative)
                if value >= 0.0:
                    intensity = min(int(255 * (value / 10.0) ** self.brightness), 255)
                    color = f"#{0:02x}{intensity:02x}{0:02x}"
                else:
                    intensity = min(int(255 * ((-value) / 10.0) ** self.brightness), 255)
                    color = f"#{intensity:02x}{0:02x}{0:02x}"

                # Draw cell
                x1, y1 = x * self.sqsize, y * self.sqsize
                x2, y2 = (x + 1) * self.sqsize, (y + 1) * self.sqsize
                self.create_rectangle(x1, y1, x2, y2, fill=color, outline='white')

                # Draw optimal action arrow
                self.draw_optimal_action(x, y)

                # Draw value text
                value_text = f"{value:.2f}"
                self.create_text(x1 + self.sqsize//2, y1 + self.sqsize//2,
                               text=value_text, fill='white', font=('Arial', self.font_size))

        # Draw grid lines
        self.draw_grid_lines()

    def draw_optimal_action(self, x: int, y: int):
        """Draw arrow indicating optimal action"""
        center_x = x * self.sqsize + self.sqsize // 2
        center_y = y * self.sqsize + self.sqsize // 2

        # Find optimal action (action with highest Q-value)
        qvals = self.vi_core.qvalues[x][y]
        optimal_action = qvals.index(max(qvals))

        # Draw arrow based on optimal action
        if optimal_action == 0:  # Up
            points = [center_x - self.twid, center_y,
                     center_x + self.twid, center_y,
                     center_x, y * self.sqsize + 5]
        elif optimal_action == 1:  # Right
            points = [center_x, center_y - self.twid,
                     center_x, center_y + self.twid,
                     (x + 1) * self.sqsize - 5, center_y]
        elif optimal_action == 2:  # Down
            points = [center_x - self.twid, center_y,
                     center_x + self.twid, center_y,
                     center_x, (y + 1) * self.sqsize - 5]
        else:  # Left
            points = [center_x, center_y - self.twid,
                     center_x, center_y + self.twid,
                     x * self.sqsize + 5, center_y]

        self.create_polygon(points, fill='blue')

    def draw_grid_lines(self):
        """Draw the grid lines"""
        # Outer boundary
        self.create_rectangle(0, 0, 10 * self.sqsize, 10 * self.sqsize,
                            outline='blue', width=2, fill='')

        # Internal grid lines
        for i in range(1, 10):
            # Vertical lines
            self.create_line(i * self.sqsize, 0, i * self.sqsize, 10 * self.sqsize,
                           fill='white', width=1)
            # Horizontal lines
            self.create_line(0, i * self.sqsize, 10 * self.sqsize, i * self.sqsize,
                           fill='white', width=1)


class VIGui:
    """Main application class - Python equivalent of VIgui.java"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Value Iteration Visualization")
        self.root.geometry("800x600")

        # Initialize core
        self.core = VICore()

        # GUI variables
        self.discount_var = tk.StringVar(value=str(self.core.discount))
        self.initial_value_var = tk.StringVar(value="0.0")
        self.absorbing_var = tk.BooleanVar(value=False)

        self.setup_gui()

        # Initial display
        self.grid_panel.update_display()

    def setup_gui(self):
        """Set up the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for grid
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Grid with scrollbars
        self.grid_panel = GridPanel(left_frame, self.core)

        # Scrollable canvas container
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)

        self.grid_panel.configure(scrollregion=self.grid_panel.bbox("all"),
                                 xscrollcommand=h_scroll.set,
                                 yscrollcommand=v_scroll.set)

        h_scroll.configure(command=self.grid_panel.xview)
        v_scroll.configure(command=self.grid_panel.yview)

        self.grid_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Right panel for controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self.setup_controls(right_frame)

    def setup_controls(self, parent):
        """Set up control buttons and inputs"""
        # Step button
        step_frame = ttk.Frame(parent)
        step_frame.pack(pady=10)

        step_btn = ttk.Button(step_frame, text="Step", command=self.do_step)
        step_btn.pack()

        # Discount controls
        discount_frame = ttk.LabelFrame(parent, text="Discount")
        discount_frame.pack(pady=10, fill=tk.X)

        discount_control_frame = ttk.Frame(discount_frame)
        discount_control_frame.pack(pady=5)

        ttk.Button(discount_control_frame, text="-", width=3,
                  command=lambda: self.adjust_discount(-0.1)).pack(side=tk.LEFT)

        discount_entry = ttk.Entry(discount_control_frame, textvariable=self.discount_var, width=8)
        discount_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(discount_control_frame, text="+", width=3,
                  command=lambda: self.adjust_discount(0.1)).pack(side=tk.LEFT)

        # Reset controls
        reset_frame = ttk.Frame(parent)
        reset_frame.pack(pady=10)

        reset_btn = ttk.Button(reset_frame, text="Reset", command=self.do_reset)
        reset_btn.pack()

        # Initial value
        initial_frame = ttk.LabelFrame(parent, text="Initial Value")
        initial_frame.pack(pady=10, fill=tk.X)

        initial_entry = ttk.Entry(initial_frame, textvariable=self.initial_value_var, width=10)
        initial_entry.pack(pady=5)

        # Display controls
        display_frame = ttk.LabelFrame(parent, text="Display")
        display_frame.pack(pady=10, fill=tk.X)

        # Brightness
        brightness_frame = ttk.Frame(display_frame)
        brightness_frame.pack(pady=5)
        ttk.Label(brightness_frame, text="Brightness").pack()

        brightness_control = ttk.Frame(brightness_frame)
        brightness_control.pack()

        ttk.Button(brightness_control, text="-", width=3,
                  command=lambda: self.adjust_brightness(1.1)).pack(side=tk.LEFT)
        ttk.Button(brightness_control, text="0", width=3,
                  command=lambda: self.reset_brightness()).pack(side=tk.LEFT)
        ttk.Button(brightness_control, text="+", width=3,
                  command=lambda: self.adjust_brightness(1/1.1)).pack(side=tk.LEFT)

        # Font size
        font_frame = ttk.Frame(display_frame)
        font_frame.pack(pady=5)
        ttk.Label(font_frame, text="Font Size").pack()

        font_control = ttk.Frame(font_frame)
        font_control.pack()

        ttk.Button(font_control, text="-", width=3,
                  command=lambda: self.adjust_font(-1)).pack(side=tk.LEFT)
        ttk.Button(font_control, text="+", width=3,
                  command=lambda: self.adjust_font(1)).pack(side=tk.LEFT)

        # Grid size
        grid_frame = ttk.Frame(display_frame)
        grid_frame.pack(pady=5)
        ttk.Label(grid_frame, text="Grid Size").pack()

        grid_control = ttk.Frame(grid_frame)
        grid_control.pack()

        ttk.Button(grid_control, text="-", width=3,
                  command=lambda: self.adjust_grid_size(-5)).pack(side=tk.LEFT)
        ttk.Button(grid_control, text="+", width=3,
                  command=lambda: self.adjust_grid_size(5)).pack(side=tk.LEFT)

        # Absorbing states checkbox
        absorbing_frame = ttk.Frame(parent)
        absorbing_frame.pack(pady=10)

        absorbing_check = ttk.Checkbutton(absorbing_frame, text="Absorbing States",
                                        variable=self.absorbing_var,
                                        command=self.toggle_absorbing)
        absorbing_check.pack()

    def do_step(self):
        """Perform one step of value iteration"""
        try:
            discount = float(self.discount_var.get())
            self.core.dostep(discount)
            self.grid_panel.update_display()
        except ValueError:
            pass  # Invalid discount value, ignore

    def do_reset(self):
        """Reset the value iteration"""
        try:
            initial_value = float(self.initial_value_var.get())
            self.core.doreset(initial_value)
            self.grid_panel.update_display()
        except ValueError:
            pass  # Invalid initial value, ignore

    def adjust_discount(self, delta: float):
        """Adjust discount factor"""
        try:
            current = float(self.discount_var.get())
            new_value = max(0.0, min(1.0, current + delta))  # Clamp between 0 and 1
            self.discount_var.set(f"{new_value:.2f}")
            self.grid_panel.update_display()
        except ValueError:
            pass

    def adjust_brightness(self, factor: float):
        """Adjust brightness"""
        self.grid_panel.brightness *= factor
        self.grid_panel.update_display()

    def reset_brightness(self):
        """Reset brightness to default"""
        self.grid_panel.brightness = 1.0
        self.grid_panel.update_display()

    def adjust_font(self, delta: int):
        """Adjust font size"""
        self.grid_panel.font_size = max(8, self.grid_panel.font_size + delta)
        self.grid_panel.update_display()

    def adjust_grid_size(self, delta: int):
        """Adjust grid cell size"""
        self.grid_panel.sqsize = max(20, self.grid_panel.sqsize + delta)
        self.grid_panel.configure(width=self.grid_panel.sqsize * 10,
                                 height=self.grid_panel.sqsize * 10)
        self.grid_panel.update_display()

    def toggle_absorbing(self):
        """Toggle absorbing states"""
        self.core.absorbing = self.absorbing_var.get()
        self.grid_panel.update_display()

    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = VIGui()
    app.run()