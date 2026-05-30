import numpy as np
import matplotlib.pyplot as plt

class WindFarm:
    """Analytical Wind Farm Wake Effect simulator.

    Calculates turbine wake interference and velocity deficits using the
    Bastankhah wake model combined with Katic quadratic superposition.
    """
    def __init__(self, turbine_coords, wind_direction=270.0,
                 wind_speed_free_stream=10.59,
                 turbine_diameter=240.0, wake_k=0.0324555, ct_coeff=8/9):
        self.original_positions = np.array(turbine_coords)
        self.n_turbines = len(turbine_coords)
        self.wind_direction = wind_direction
        self.U_inf = wind_speed_free_stream
        self.D = turbine_diameter
        self.k_y = wake_k
        self.CT = ct_coeff
        self.turbine_velocities = {}

    def _rotate_coordinates(self, coords):
        """Rotate the coordinate system to align the wind direction with the X-axis."""
        angle_rad = np.radians(self.wind_direction - 270.0)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_rot = coords[:, 0] * cos_a - coords[:, 1] * sin_a
        y_rot = coords[:, 0] * sin_a + coords[:, 1] * cos_a
        return np.column_stack((x_rot, y_rot))

    def _bastankhah_wake_deficit(self, x, y, x0, y0):
        """Compute the velocity deficit from a single turbine at (x0, y0)."""
        d = x - x0
        is_downstream = d > 0
        sigma_y = np.full_like(d, np.inf)
        sigma_y[is_downstream] = self.k_y * d[is_downstream] + self.D / np.sqrt(8)
        radical_term = 1.0 - self.CT / (8.0 * (sigma_y / self.D)**2)
        radical_term = np.maximum(0.0, radical_term)
        exponent = -0.5 * ((y - y0) / sigma_y)**2
        deficit = (1.0 - np.sqrt(radical_term)) * np.exp(exponent)
        return np.where(is_downstream, deficit, 0.0)

    def calculate_wake_effects(self):
        """Compute effective wind velocity at each turbine using quadratic superposition."""
        rotated_turbine_pos = self._rotate_coordinates(self.original_positions)
        velocities = []
        for i in range(self.n_turbines):
            xi, yi = rotated_turbine_pos[i]
            sum_deficits_sq = 0.0
            for j in range(self.n_turbines):
                if i == j:
                    continue
                xj, yj = rotated_turbine_pos[j]
                deficit = self._bastankhah_wake_deficit(xi, yi, xj, yj)
                sum_deficits_sq += deficit**2
            total_deficit = min(np.sqrt(sum_deficits_sq), 1.0)
            velocities.append(self.U_inf * (1.0 - total_deficit))
        self.turbine_velocities = dict(zip(map(tuple, self.original_positions),
                                           velocities))

    def get_velocity_field(self, resolution=300, x_bounds=None, y_bounds=None):
        """Generate a grid mesh and calculate wind velocity at each grid point."""
        if x_bounds is not None and y_bounds is not None:
            x_min, x_max = x_bounds
            y_min, y_max = y_bounds
        else:
            buffer = self.D * 4
            x_min, y_min = self.original_positions.min(axis=0) - buffer
            x_max, y_max = self.original_positions.max(axis=0) + buffer
        grid_x = np.linspace(x_min, x_max, resolution)
        grid_y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(grid_x, grid_y)
        grid_coords = np.vstack([X.ravel(), Y.ravel()]).T
        rotated_grid_coords = self._rotate_coordinates(grid_coords)
        rotated_turbine_pos = self._rotate_coordinates(self.original_positions)
        total_deficit_sq = np.zeros(resolution * resolution)
        for xj_rot, yj_rot in rotated_turbine_pos:
            deficit = self._bastankhah_wake_deficit(
                rotated_grid_coords[:, 0], rotated_grid_coords[:, 1],
                xj_rot, yj_rot)
            total_deficit_sq += deficit**2
        velocity_field = self.U_inf * (1.0 - np.sqrt(np.minimum(total_deficit_sq, 1.0)))
        return X, Y, velocity_field.reshape(resolution, resolution)

    def summarize_results(self):
        """Print a summary table of wind speeds at each turbine location."""
        if not self.turbine_velocities:
            self.calculate_wake_effects()
        print("-" * 75)
        print(f"Wind direction: {self.wind_direction}° | Freestream velocity: {self.U_inf:.2f} m/s")
        print("-" * 75)
        print(f"{'Turbine #':<12} {'Position (x, y)':<25} {'Velocity (m/s)':<20}")
        print("-" * 75)
        for i, pos in enumerate(self.original_positions):
            vel = self.turbine_velocities[tuple(pos)]
            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})"
            print(f"T{i+1:<11} {pos_str:<25} {vel:<20.4f}")
        print("-" * 75)

    def _draw_turbine(self, ax, x_pos, y_pos, turbine_id=None, velocity=None,
                      with_text=False, fontsize=8):
        """Draw a turbine rotor symbol on the plot."""
        rotor_angle_rad = np.radians(270 - self.wind_direction + 90)
        radius = self.D / 2

        dx = radius * np.cos(rotor_angle_rad)
        dy = radius * np.sin(rotor_angle_rad)

        ax.plot([x_pos - dx, x_pos + dx], [y_pos - dy, y_pos + dy],
                color='black', linewidth=2.5, zorder=6)

        ax.scatter(x_pos, y_pos, s=30, c='white', edgecolor='black', zorder=7)

        if with_text and turbine_id is not None and velocity is not None:
            label_text = f"T{turbine_id}\n{velocity:.2f} m/s"
            offset_angle_rad = np.radians(270 - self.wind_direction + 90)
            offset_distance = radius * 1.3
            offset_x = offset_distance * np.cos(offset_angle_rad)
            offset_y = offset_distance * np.sin(offset_angle_rad)
            ha = 'left' if offset_x >= 0 else 'right'
            va = 'bottom' if offset_y >= 0 else 'top'
            ax.text(x_pos + offset_x, y_pos + offset_y, label_text, ha=ha,
                    va=va, fontsize=fontsize, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.6,
                              edgecolor='none', pad=1), zorder=8)

    def plot_layout_with_wake_field(self, title=None, save_path=None):
        """Plot the wind velocity field under wake effects."""
        if not self.turbine_velocities:
            self.calculate_wake_effects()

        # ==========================================
        # PLOT STYLE CONTROL PANEL
        # Manage all font sizes here
        # ==========================================
        base = 30
        font_config = {
            'title': base * 1,
            'axis_label': base * 0.9,
            'tick_label': base * 0.7,
            'turbine_label': 13,  # Legend T1, T2...
            'wind_label': base * 0.6,
            'cbar_label': base * 0.5
        }
        # ==========================================

        X, Y, V = self.get_velocity_field()

        x_range = X.max() - X.min()
        y_range = Y.max() - Y.min()
        aspect_ratio = y_range / x_range if x_range != 0 else 1

        fig_width = 14
        fig_height = fig_width * aspect_ratio
        if fig_height > 14:
            fig_height = 14
            fig_width = fig_height / aspect_ratio

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        cmap_reversed = 'coolwarm_r'
        contour = ax.contourf(X, Y, V, levels=100, cmap=cmap_reversed, zorder=1)

        for i, pos in enumerate(self.original_positions):
            vel = self.turbine_velocities[tuple(pos)]
            self._draw_turbine(ax, pos[0], pos[1], turbine_id=i+1,
                               velocity=vel,
                               with_text=True,
                               fontsize=font_config['turbine_label'])

        flow_angle_rad = np.radians(270 - self.wind_direction)
        ax.arrow(0.05, 0.9, 0.06 * np.cos(flow_angle_rad),
                 0.06 * np.sin(flow_angle_rad),
                 transform=ax.transAxes, width=0.008,
                 head_width=0.025, head_length=0.018,
                 fc='black', ec='black', zorder=10)
        ax.text(0.05, 0.95, 'Wind', transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=font_config['wind_label'], fontweight='bold')

        if title is None:
            title_text = f'Velocity Field (Direction: {self.wind_direction}°)'
        else:
            title_text = title
        ax.set_title(title_text, fontsize=font_config['title'], weight='bold')

        ax.set_xlabel('X Coordinate [m]', fontsize=font_config['axis_label'])
        ax.set_ylabel('Y Coordinate [m]', fontsize=font_config['axis_label'])
        ax.tick_params(axis='both', which='major',
                       labelsize=font_config['tick_label'])

        ax.set_aspect('equal', adjustable='box')

        cbar = fig.colorbar(contour, ax=ax, orientation='vertical',
                            pad=0.02, shrink=0.855)
        cbar.set_label('Wind Speed [m/s]',
                       fontsize=font_config['cbar_label'], weight='bold')
        cbar.ax.tick_params(labelsize=font_config['tick_label'])

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

if __name__ == "__main__":
    # Quick standalone testing example
    test_coordinates = [
        [0.0, 0.0],
        [300.0, 0.0],
        [0.0, 300.0],
        [300.0, 300.0]
    ]

    farm = WindFarm(test_coordinates, wind_direction=270.0,
                    turbine_diameter=130.0,
                    wind_speed_free_stream=10.0)

    farm.summarize_results()