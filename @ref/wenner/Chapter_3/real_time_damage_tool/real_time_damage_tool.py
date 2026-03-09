"""
This file was created by Frankie Iovinelli as a prototype for the working instantaneous damage model.
This serves only as a prototype for the working model, and demands various improvements as outlined in his report, particularly making it function on a transient basis.
However, it is still useful as it visualizes the workflow and the necessary steps that should be used in the final version.

last modified: 6/2/2025
Built by: Frankie Iovinelli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import numpy.ma as ma

class Node:
    def __init__(self, node_id, tf, dt):
        self.node_id = node_id
        self.tf = tf
        self.dt = dt
        self.current_residual = 0
        self.max_residual = 0
        self.elastic_stress = 0  # Store the elastic stress for later use
        self.total_damage = 0.0  # Cumulative damage
        self.state_of_health = 1.0  # 1 - total_damage
        self.lifetime = None  # Lifetime in years

    def update_residual(self, new_residual):
        if new_residual > self.max_residual:
            self.max_residual = new_residual
            self.current_residual = new_residual

    def calculate_current_stress(self, elastic_stress, m,b):
        self.elastic_stress = elastic_stress  # Store the elastic stress
        return elastic_stress - m * self.current_residual - b

    def update_lifetime(self, instant_damage):
        # Avoid division by zero
        if instant_damage > 0:
            seconds = self.state_of_health / instant_damage
            self.lifetime = seconds / (3600 * 24 * 365.25)  # Convert to years
        else:
            self.lifetime = np.inf

class StressLookup:
    def __init__(self):
        self.elastic_table = None
        self.residual_table = None

    def load_csv(self, elastic_path, residual_path):
        self.elastic_table = pd.read_csv(elastic_path)
        self.residual_table = pd.read_csv(residual_path)

    def interpolate_stress(self, tf, dt, table):
        # Get the stress column name
        stress_column = 'Max_Elastic_Stress' if 'Max_Elastic_Stress' in table.columns else 'Max_Residual_Stress'
        
        # Find all valid points in the table
        valid_points = table[['TF', 'DT', stress_column]].values
        
        if len(valid_points) == 0:
            print(f"Warning: No valid points found in table")
            return None
            
        # Find the nearest points (up to 4) for interpolation
        distances = []
        for point in valid_points:
            # Calculate normalized distance (to handle different scales of TF and DT)
            tf_dist = abs(point[0] - tf) / (table['TF'].max() - table['TF'].min())
            dt_dist = abs(point[1] - dt) / (table['DT'].max() - table['DT'].min())
            total_dist = np.sqrt(tf_dist**2 + dt_dist**2)
            distances.append((total_dist, point))
        
        # Sort by distance and get up to 4 nearest points
        distances.sort(key=lambda x: x[0])
        nearest_points = [point for _, point in distances[:4]]
        
        if len(nearest_points) == 1:
            # Only one point - use nearest neighbor
            return nearest_points[0][2]
        elif len(nearest_points) == 2:
            # Two points - use linear interpolation
            p1, p2 = nearest_points
            # Calculate weights based on distance
            total_dist = distances[0][0] + distances[1][0]
            w1 = distances[1][0] / total_dist
            w2 = distances[0][0] / total_dist
            return w1 * p1[2] + w2 * p2[2]
        else:
            # Three or four points - use inverse distance weighting
            weights = []
            values = []
            for point in nearest_points:
                # Calculate normalized distance
                tf_dist = abs(point[0] - tf) / (table['TF'].max() - table['TF'].min())
                dt_dist = abs(point[1] - dt) / (table['DT'].max() - table['DT'].min())
                dist = np.sqrt(tf_dist**2 + dt_dist**2)
                if dist == 0:  # Exact match
                    return point[2]
                weight = 1.0 / (dist + 1e-10)  # Add small epsilon to avoid division by zero
                weights.append(weight)
                values.append(point[2])
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
            
            # Calculate weighted average
            return sum(w * v for w, v in zip(weights, values))

class StressModel:
    def __init__(self, num_columns):
        self.nodes = []
        self.lookup = StressLookup()
        self.num_columns = num_columns
        self.failed_cases = []

    def load_nodes(self, csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            node_id = row['node_id']
            tf = row['tf']
            dt = row['dt']
            self.nodes.append(Node(node_id, tf, dt))

    def process_nodes(self, m, b):
        for node in self.nodes:
            elastic_stress = self.lookup.interpolate_stress(node.tf, node.dt, self.lookup.elastic_table)
            residual_stress = self.lookup.interpolate_stress(node.tf, node.dt, self.lookup.residual_table)
            
            if elastic_stress is None or residual_stress is None:
                print(f"Warning: Could not interpolate stress for node {node.node_id}")
                self.failed_cases.append(node.node_id)
                continue
                
            node.update_residual(residual_stress)
            node.calculate_current_stress(elastic_stress, m, b)

    def output_results(self, output_path, m, b):
        results = []
        for node in self.nodes:
            # Adjust for 1-based node_id
            adjusted_id = node.node_id - 1
            x = adjusted_id % self.num_columns
            y = adjusted_id // self.num_columns
            results.append({
                'node_id': node.node_id,
                'x': x,
                'y': y,
                'stress': node.calculate_current_stress(node.elastic_stress, m, b)
            })
        pd.DataFrame(results).to_csv(output_path, index=False)

    def visualize_stress(self, output_path='stress_visualization.png'):
        plt.rcParams['font.size'] = 20

        # Read the output CSV
        df = pd.read_csv('output.csv')
        
        # Create the figure
        plt.figure(figsize=(8, 8))  # Make it square for a grid
        
        # Create a regular grid for the color mesh
        x = np.linspace(df['x'].min(), df['x'].max(), 200)
        y = np.linspace(df['y'].min(), df['y'].max(), 200)
        X, Y = np.meshgrid(x, y)
        
        # Interpolate stress values onto the regular grid
        points = df[['x', 'y']].values
        values = df['stress'].values
        Z = griddata(points, values, (X, Y), method='cubic')
        
        # Create the color mesh and keep the returned object for the colorbar
        mesh = plt.pcolormesh(X, Y, Z, cmap='plasma', shading='auto')
        
        # Add uniform black dots for node positions
        plt.scatter(df['x'], df['y'], color='black', s=20, edgecolor='none')
        
        # Set the title
        # plt.title('Current Receiver Stress')
        
        # Set axis limits exactly to the data range
        xmin, xmax = df['x'].min(), df['x'].max()
        ymin, ymax = df['y'].min(), df['y'].max()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        
        # Add a thin black border around the data
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                          fill=False, edgecolor='black', linewidth=1))
        
        # Add colorbar for the mesh
        cbar = plt.colorbar(mesh, fraction=0.046, pad=0.04)
        cbar.set_label('Stress [MPa]')
        
        # Set axis labels
        plt.xlabel('theta')
        plt.ylabel('y')
        
        # Remove axis ticks for a clean look
        plt.xticks([])
        plt.yticks([])
        
        # Remove all padding and margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save the plot with high DPI and no padding
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()

    def visualize_total_temperature(self, csv_path='receiver_info.csv', output_path='total_temperature_visualization.png'):
        plt.rcParams['font.size'] = 20

        df = pd.read_csv(csv_path)
        # Calculate total temperature
        df['total_temp'] = df['tf'] + df['dt']
        # Assume node_id is 1-based and grid is the same as before
        adjusted_id = df['node_id'] - 1
        num_columns = self.num_columns
        df['x'] = adjusted_id % num_columns
        df['y'] = adjusted_id // num_columns

        # Create the figure
        plt.figure(figsize=(8, 8))

        # Create a regular grid for the color mesh
        x = np.linspace(df['x'].min(), df['x'].max(), 200)
        y = np.linspace(df['y'].min(), df['y'].max(), 200)
        X, Y = np.meshgrid(x, y)

        # Interpolate values onto the regular grid
        points = df[['x', 'y']].values
        values = df['total_temp'].values
        Z = griddata(points, values, (X, Y), method='cubic')

        # Create the color mesh and keep the returned object for the colorbar
        mesh = plt.pcolormesh(X, Y, Z, cmap='plasma', shading='auto')

        # Add uniform black dots for node positions
        plt.scatter(df['x'], df['y'], color='black', s=20, edgecolor='none')

        # Set the title
        # plt.title('Total Temperature (Fluid + ΔT)')

        # Set axis limits exactly to the data range
        xmin, xmax = df['x'].min(), df['x'].max()
        ymin, ymax = df['y'].min(), df['y'].max()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # Add a thin black border around the data
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                          fill=False, edgecolor='black', linewidth=1))

        # Add colorbar for the mesh
        cbar = plt.colorbar(mesh, fraction=0.046, pad=0.04)
        cbar.set_label('Total Temperature [C]')

        # Set axis labels
        plt.xlabel('theta')
        plt.ylabel('y')

        # Remove axis ticks for a clean look
        plt.xticks([])
        plt.yticks([])

        # Remove all padding and margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the plot with high DPI and no padding
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()

    def visualize_lookup_table(self, table_path, value_column, output_path='lookup_table_visualization.png', title=None, colorbar_label=None):
        df = pd.read_csv(table_path)
        # Pivot the table to create a 2D grid for plotting (DT on x, TF on y)
        grid = df.pivot(index='TF', columns='DT', values=value_column)
        X, Y = np.meshgrid(grid.columns.values, grid.index.values)
        Z = grid.values

        plt.figure(figsize=(8, 6))
        mesh = plt.pcolormesh(X, Y, Z, cmap='plasma', shading='auto')
        cbar = plt.colorbar(mesh, fraction=0.046, pad=0.04)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)
        else:
            cbar.set_label(value_column)
        if title is not None:
            plt.title(title)
        else:
            plt.title(f"{value_column} Lookup Table")
        plt.xlabel('DT')
        plt.ylabel('TF')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def instantaneous_dmg(self, stress, temperature):
        """
        Calculating the damage on a 1-second rate basis
        """
        # sigmas and Ts are arrays of the same length (one per node)
        T_K = temperature + 273.15
        B0 = -26.27
        B1 = 44158
        B2 = 4.72
        B3 = -11337
        tRs = 10 ** (B0 + (B1 / T_K) + np.log10(stress ** B2) + np.log10(stress ** (B3 / T_K)))
        dt = (1/3600) # = 1 second
        time_dmg = dt / tRs
        return time_dmg

    def visualize_damage(self, output_path='damage_visualization.png'):
        plt.rcParams['font.size'] = 20

        # Read the output CSV (for stress and node positions)
        df = pd.read_csv('output.csv')
        # Read receiver_info.csv for temperature
        receiver_df = pd.read_csv('receiver_info.csv')
        # Calculate total temperature (tf + dt)
        Ts = receiver_df['tf'].values + receiver_df['dt'].values
        sigmas = df['stress'].values
        # Calculate damage
        damage = self.instantaneous_dmg(sigmas, Ts)
        df['damage'] = damage

        # Avoid log(0) by setting a small minimum value
        min_damage = np.min(damage[damage > 0])
        damage[damage <= 0] = min_damage
        log_damage = np.log10(damage)

        # Create the figure
        plt.figure(figsize=(8, 8))

        # Create a regular grid for the color mesh
        x = np.linspace(df['x'].min(), df['x'].max(), 200)
        y = np.linspace(df['y'].min(), df['y'].max(), 200)
        X, Y = np.meshgrid(x, y)

        # Interpolate log-damage values onto the regular grid
        points = df[['x', 'y']].values
        Z_log = griddata(points, log_damage, (X, Y), method='linear')
        Z = np.power(10, Z_log)

        # Set up log normalization for colorbar
        norm = mcolors.LogNorm(vmin=min_damage, vmax=damage.max())

        # Create the color mesh and keep the returned object for the colorbar
        mesh = plt.pcolormesh(X, Y, Z, cmap='plasma', shading='auto', norm=norm)

        # Add uniform black dots for node positions
        plt.scatter(df['x'], df['y'], color='black', s=20, edgecolor='none')

        # Set the title
        # plt.title('Instantaneous Damage (log scale)')

        # Set axis limits exactly to the data range
        xmin, xmax = df['x'].min(), df['x'].max()
        ymin, ymax = df['y'].min(), df['y'].max()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # Add a thin black border around the data
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                          fill=False, edgecolor='black', linewidth=1))

        # Add colorbar for the mesh
        cbar = plt.colorbar(mesh, fraction=0.046, pad=0.04)
        cbar.set_label('Damage (1/s, log scale)')

        # Set axis labels
        plt.xlabel('theta')
        plt.ylabel('y')

        # Remove axis ticks for a clean look
        plt.xticks([])
        plt.yticks([])

        # Remove all padding and margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the plot with high DPI and no padding
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()

    def update_all_lifetimes(self):
        # Read the output CSV (for stress and node positions)
        df = pd.read_csv('output.csv')
        receiver_df = pd.read_csv('receiver_info.csv')
        Ts = receiver_df['tf'].values + receiver_df['dt'].values
        sigmas = df['stress'].values
        instant_damage = self.instantaneous_dmg(sigmas, Ts)
        # Update each node
        for node, dmg in zip(self.nodes, instant_damage):
            node.update_lifetime(dmg)

    def visualize_lifetime(self, output_path='lifetime_visualization.png'):
        plt.rcParams['font.size'] = 20
        # Read the output CSV (for node positions)
        df = pd.read_csv('output.csv')
        # Read receiver_info.csv for temperature
        receiver_df = pd.read_csv('receiver_info.csv')
        Ts = receiver_df['tf'].values + receiver_df['dt'].values
        sigmas = df['stress'].values
        # Calculate instantaneous damage
        instant_damage = self.instantaneous_dmg(sigmas, Ts)
        # For now, assume state_of_health is 1 for all nodes (unless you update it elsewhere)
        state_of_health = np.ones_like(instant_damage)
        # If you want to use actual state_of_health, you can pass it in or update it before this call
        lifetime_sec = state_of_health / instant_damage
        lifetime_years = lifetime_sec / (3600 * 24 * 365.25)
        df['lifetime_years'] = lifetime_years

        # Avoid log(0) by setting a small minimum value
        min_lifetime = np.min(lifetime_years[lifetime_years > 0])
        lifetime_years[lifetime_years <= 0] = min_lifetime
        log_values = np.log10(lifetime_years)

        # Create the figure
        plt.figure(figsize=(8, 8))

        # Create a regular grid for the color mesh
        x = np.linspace(df['x'].min(), df['x'].max(), 200)
        y = np.linspace(df['y'].min(), df['y'].max(), 200)
        X, Y = np.meshgrid(x, y)

        # Interpolate log-lifetime values onto the regular grid
        points = df[['x', 'y']].values
        Z_log = griddata(points, log_values, (X, Y), method='linear')
        Z = np.power(10, Z_log)

        # Set up log normalization for colorbar
        norm = mcolors.LogNorm(vmin=min_lifetime, vmax=lifetime_years.max())

        # Create the color mesh and keep the returned object for the colorbar
        mesh = plt.pcolormesh(X, Y, Z, cmap='plasma', shading='auto', norm=norm)

        # Add uniform black dots for node positions
        plt.scatter(df['x'], df['y'], color='black', s=20, edgecolor='none')

        # Set the title
        # plt.title('Estimated Lifetime (years, log scale)')

        # Set axis limits exactly to the data range
        xmin, xmax = df['x'].min(), df['x'].max()
        ymin, ymax = df['y'].min(), df['y'].max()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # Add a thin black border around the data
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                          fill=False, edgecolor='black', linewidth=1))

        # Add colorbar for the mesh
        cbar = plt.colorbar(mesh, fraction=0.046, pad=0.04)
        cbar.set_label('Lifetime [years] (log scale)')

        # Set axis labels
        plt.xlabel('theta')
        plt.ylabel('y')

        # Remove axis ticks for a clean look
        plt.xticks([])
        plt.yticks([])

        # Remove all padding and margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the plot with high DPI and no padding
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()

# Example usage
# TODO: pass 1 matrix of tf and 1 matrix of flux
model = StressModel(num_columns=10)
model.lookup.load_csv('elastic_stress_lookup_table.csv', 'residual_stress_lookup_table.csv')
model.load_nodes('receiver_info.csv')
model.process_nodes(m=0.9337,b=-3.5)
model.output_results('output.csv',m=0.9337,b=-3.5)
model.visualize_stress()  # This will create 'stress_visualization.png'
model.visualize_total_temperature()  # This will create 'total_temperature_visualization.png'
model.visualize_lookup_table(
    'elastic_stress_lookup_table.csv',
    value_column='Max_Elastic_Stress',
    output_path='elastic_lookup_visualization.png',
    title='Elastic Stress Lookup Table',
    colorbar_label='Elastic Stress [MPa]'
)
model.visualize_lookup_table(
    'residual_stress_lookup_table.csv',
    value_column='Max_Residual_Stress',
    output_path='residual_lookup_visualization.png',
    title='Residual Stress Lookup Table',
    colorbar_label='Residual Stress [MPa]'
)
model.visualize_damage()  # This will create 'damage_visualization.png'
model.visualize_lifetime()  # This will create 'lifetime_visualization.png'

print(f"Number of failed cases: {len(model.failed_cases)}")
print(f"Number of total cases: {len(model.nodes)}")
