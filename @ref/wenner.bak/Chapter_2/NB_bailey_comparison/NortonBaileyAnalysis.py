"""
This file was the primary file used by Frankie Iovinelli to
1. Develop and test the Modified Norton-Bailey model as detailed on Page 33-41 of his report
2. Assess how many hours of a particular cycle are responsible for x% of the total daily damage
3. Build version 1 of his instantaneous damage model: a version that simply interpolates damage in the DT TF space using peak FEA damage on the 20th day of that DT TF case (overly conservative)
4. Assess the difference in damage between 20 days using Norton-Bailey and 20 days using FEA - a plot made for Jacob Wenner 

*Note: This file contains no necessary information with regard to the final version of the instantaneous damage model
last modified: 6/2/2025
Built by: Frankie Iovinelli
"""

import matplotlib
# matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import LifetimeCalculations as lc
import BoundaryConditions as bc
import NortonBailey as nb
import matplotlib.colors as mcolors
# import plotly.express as px
# import plotly.graph_objects as go
from scipy.interpolate import interp2d, RegularGridInterpolator, RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D
import processingFuns

#! SETTINGS FOR THE CODE
# Do you want to see the detailed TF DT plots containing stress evolution, NB predictions, NB accuracy, and lifetime accuracy over time?
detailed_plots = False

# Do you want to perform an error analysis?
error_analysis = False

#! Obtaining data from the CSV files

material = "A230" # or 617
if(material == "A230"):
    base_path = './20_Day_Plastic_Stress_Data_A230/'
elif(material == "A617"):
    base_path = './20_Day_Plastic_Stress_Data_A617/'
else:
    print("No material selected")
    exit()


# Find CSV files
path = os.path.join(base_path, "*.csv")
print(f"\nLooking for CSV files in: {path}")
csv_files = glob.glob(path)
# Filter CSV files to only include those with "stresses" in the filename
csv_files = [f for f in csv_files if "stresses" in f.lower()]
print(f"Found {len(csv_files)} CSV files with 'stresses' in the name")

# Create dictionary to store the data
stress_data = {}

# Create sets to track files
processed_files = set()
failed_files = set()
duplicate_ids = set()

#! File Processing
print("\nProcessing files:")
for file in csv_files:
    filename = os.path.basename(file)
    
    try:
        # Find positions of 'tf' and 'td' in filename
        tf_pos = filename.find('Tf')
        td_pos = filename.find('dT')
        substep_pos = filename.find('substep')
        period_pos = filename.find('period')
        
        
        # Get 3 digits before each identifier
        tf_num = filename[tf_pos-3:tf_pos]
        td_num = filename[td_pos-3:td_pos]
        substep_num = filename[substep_pos-2:substep_pos]
        period_num = filename[period_pos-2:period_pos]
        
        
        # Create the ID
        file_id = f"{tf_num}TF_{td_num}TD_{substep_num}SUB_{period_num}PER"
        
        # Check if this ID already exists
        if file_id in stress_data:
            duplicate_ids.add(file_id)
            print(f"WARNING: Duplicate ID found: {file_id}")
            print(f"  Original file: {processed_files[file_id]}")
            print(f"  New file: {filename}")
    
        # Read the CSV and store data
        df = pd.read_csv(file)
        stress_data[file_id] = df
        processed_files.add(filename)
        
        print(f"Processed: {filename} → {file_id}")
        
    except Exception as e:
        failed_files.add(filename)
        print(f"Error processing {filename}: {e}")

print(f"\nSummary:")
print(f"Total CSV files found: {len(csv_files)}")
print(f"Successfully processed: {len(processed_files)}")
print(f"Failed to process: {len(failed_files)}")
print(f"Duplicate IDs found: {len(duplicate_ids)}")

if failed_files:
    print("\nFailed files:")
    for f in failed_files:
        print(f"  {f}")

if duplicate_ids:
    print("\nDuplicate IDs:")
    for id in duplicate_ids:
        print(f"  {id}")


#! Determine representitive temperature for each day
def get_Youngs_Modulus(T, material):
    """
    Get the young's modulus from the deformation model and a temperature (copied from get function on SR Life computer)
    """
    #get young's modulus
    if(material == "A230"):
        E_temps = [298.15, 373.15, 473.15, 573.15, 673.15, 773.15, 873.15, 973.15, 1073.15, 1173.15] #! hardcoded
        E = [211000, 206000, 200000, 195000, 189000, 183000, 176000, 168000, 159000, 149000] #MPa #! hardcoded
        Ecycle = np.interp(T, E_temps, E)
    elif(material == "A617"):
        E_temps =[273.15, 288.74183673,  304.33367347,  319.9255102,   335.51734694,  351.10918367,  366.70102041,  382.29285714,  397.88469388,  413.47653061,  429.06836735,  444.66020408,  460.25204082,  475.84387755,  491.43571429,  507.02755102,  522.61938776,  538.21122449,  553.80306122,  569.39489796,  584.98673469,  600.57857143,  616.17040816,  631.7622449,   647.35408163,  662.94591837,  678.5377551,   694.12959184,  709.72142857,  725.31326531,  740.90510204,  756.49693878,  772.08877551,  787.68061225,  803.27244898,  818.86428571,  834.45612245,  850.04795918,  865.63979592,  881.23163265,  896.82346939,  912.41530612,  928.00714286,  943.59897959,  959.19081633,  974.78265306,  990.3744898,  1005.96632653, 1021.55816327, 1037.15]
        E =  [202666.666667, 201627.210884, 200587.755102, 199548.29932,  198508.843537, 197469.387755, 196429.931973, 195451.428571, 194515.918367, 193580.408163, 192763.265306, 192139.591837, 191515.918367, 190892.244898, 190268.571429, 189644.897959, 189021.22449,  188397.55102,  187773.877551, 187150.204082, 186289.795918, 185354.285714, 184418.77551,  183483.265306, 182547.755102, 181612.244898, 180676.734694, 179741.22449,  178805.714286, 177826.938776, 176579.591837, 175332.244898, 174084.897959, 173128.163265, 172192.653061, 171257.142857, 170095.510204, 168848.163265, 167600.816327, 166515.102041, 165579.591837, 164644.081633, 163611.428571, 162364.081633, 161116.734694, 159869.387755, 158622.040816, 157374.693878, 156127.346939, 154880]
        Ecycle = np.interp(T, E_temps, E)
    return Ecycle

def get_Temp_Norton_Bailey_Gonzalez(sigma_eq, sigma_eq_r, t_stab, E, R = 8.314/1000, n =6.6, m = 0.00, A = 2.6880e-45, Q = 322):
    try:
        C5 = ((sigma_eq/E)**(1-n))/(1-n)
        C4 = (sigma_eq-sigma_eq_r)/E
        C3 = 1/(1-n)*C4**(1-n)-C5
        C2 = -(A*E**n)*(t_stab**(m+1)/(m+1))
        C1 = C3/C2
        if C1 <= 0:  # log of negative or zero is undefined
            #print(f"C1 is negative or zero: {C1}")
            return float('nan')
        T = -Q/R*1/np.log(C1)
        return T
    except Exception as e:
        print(f"Error in Norton-Bailey calculation: {e}")
        return float('nan')

# Initialize dictionary to store temperatures for each file
NB_temps = {file_id: [] for file_id in stress_data}
E_temps = {file_id: [] for file_id in stress_data}
file_total_temp = {file_id: 0 for file_id in stress_data}
substep_num = {file_id: 0 for file_id in stress_data}
period_num = {file_id: 0 for file_id in stress_data}
for file_id in stress_data:
    print("ANALYZING FILE: ", file_id)
    # Extract parameters from file_id
    substep_num[file_id] = int(file_id.split('_')[2].replace('SUB', ''))
    period_num[file_id] = int(file_id.split('_')[3].replace('PER', ''))
    tf_num = int(file_id.split('TF_')[0])
    td_num = int(file_id.split('TD_')[0].split('_')[1])
    total_temp = tf_num + td_num
    file_total_temp[file_id] = total_temp
    
    daily_steps = substep_num[file_id] * period_num[file_id]
    days = 20  #! hardcoded

    # Get first day max stress using 'vmRoutsAll' column
    df = stress_data[file_id]
    sigma_eq = df['vmRoutsAll'].iloc[0:daily_steps].max() * 1e6  # [Pa]
    

    # Start from day 2 (index 1) since day 1 is our reference
    for day in range(1, days):
        #print(f'\n=== Day {day} Analysis ===')
        # Calculate stress difference between first day and current day
        current_day_max = df['vmRoutsAll'].iloc[day*daily_steps:(day+1)*daily_steps].max() * 1e6
        sigma_eq_r = sigma_eq - current_day_max  # [Pa]
        
        #print(f'First day max  stress (sigma_eq): {sigma_eq/1e6:.2f} MPa')
        #print(f'Current day max  stress: {current_day_max/1e6:.2f} MPa')
        #print(f'Stress difference (sigma_eq_r): {sigma_eq_r/1e6:.2f} MPa')
        #print(f'Time (t_stab): {(day+1)*60*60*24} seconds ({day+1} days)')
        
        # Initial temperature guess
        T_guess = 500+273.15
        T_new = T_guess
        residual = 1.0
        max_iter = 100
        iter_count = 0
        
        # finding representitve temperature modulus
        while residual > 1e-6 and iter_count < max_iter:
            T_old = T_new
            youngs_modulus = get_Youngs_Modulus(T_old,material)*1e6 # [Pa]
            
            #print(f'\nIteration {iter_count + 1}:')
            #print(f'Temperature guess: {T_old-273.15:.2f}°C ({T_old:.2f}K)')
            #print(f'Young\'s modulus: {youngs_modulus/1e9:.2f} GPa')
            
            T_new = get_Temp_Norton_Bailey_Gonzalez(
                sigma_eq=sigma_eq,      #[Pa]
                sigma_eq_r=sigma_eq_r, # [Pa]
                t_stab=(day+1)*60*60,  #seconds, 
                E=youngs_modulus #[Pa]
            )
            
            # Calculate residual
            residual = abs(T_new - T_old)
            iter_count += 1
        
        # Check if temperature is NaN and handle accordingly
        if np.isnan(T_new):
            #print(f"Warning: Day {day} resulted in NaN temperature - skipping and adding 0")
            NB_temps[file_id].append(0)
            E_temps[file_id].append(0)
            continue
        
        # Store converged temperature in the dictionary list for this file_id
        NB_temps[file_id].append(T_new)
        E_temps[file_id].append(youngs_modulus)
        
        # Debugging Code:
        # if iter_count == max_iter:
        #     print(f'Warning: Day {day} did not converge after {max_iter} iterations')
        # if day == 19:
        #     print(f'The Norton Bailey representative temperature for day {day} is {T_new:.4f}°K')
        # print(f'Converged in {iter_count} iterations with residual {residual:.1e}')


    # CREATE TWO SUBPLOTS SIDE BY SIDE
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns
    
    # Temperature evolution plot
    # temps_array = np.array(NB_temps[file_id])  # Convert list to numpy array
    # ax1.plot(range(2, days+1), temps_array-273.15, label='Norton-Bailey Representative Temperature')
    # ax1.set_xlabel('Day')
    # ax1.set_ylabel('Temperature (°C)')
    # ax1.set_title(f'Temperature Evolution - {file_id}')
    # ax1.grid(True)
    # ax1.legend()

    # Max stresses plot
    # ax2.plot(df['vmRoutsAll'], label='Maximum Stresses')
    # ax2.set_xlabel('Time Step')
    # ax2.set_ylabel('Stress (MPa)')
    # ax2.set_title(f'Maximum Stresses - {file_id}')
    # ax2.grid(True)
    # ax2.legend()

    #plt.show()

if detailed_plots:
    # #! plotting NB rep temp behavior on tf dt plot to seek patterns
    # the following code was used to seek patterns in the NB rep temp behavior. the only one that is moderately useful is the correlation between the 20th day temperature divided by the max temperature

    # # Create main scatter plot
    # fig, ax = plt.subplots(figsize=(12, 10))

    # # Lists to store values
    # tf_values = []
    # dt_values = []

    # colors = []

    # # Create scatter plot first
    # for file_id in stress_data:
    #     tf_num = int(file_id.split('TF_')[0])
    #     dt_num = int(file_id.split('TD_')[0].split('_')[1])


    #     tf_values.append(tf_num)
    #     dt_values.append(dt_num)
        
    #     temps = NB_temps[file_id]
    #     if 0 in temps:
    #         colors.append('gray')
    #     elif temps[-1] < temps[0]:
    #         colors.append('red')
    #     elif temps[-1] > temps[0]:
    #         colors.append('green')

    # # Create scatter plot
    # plt.scatter(dt_values, tf_values, c=colors, s=100)
    # plt.xlabel('DT (Temperature Difference [°C])')
    # plt.ylabel('TF (Fluid Temperature [°C])')
    # plt.title('Norton-Bailey Representative Temperature Behavior: Slope')

    # # Add legend
    # legend_elements = [
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Contains Zero Temperature'),
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Temperature Decreased'),
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Temperature Increased')
    # ]
    # plt.legend(handles=legend_elements)

    # # Add dashed red line: y = -41x + 330.28
    # dt_range = np.array([min(dt_values), max(dt_values)])  # Create x-range for line
    # tf_line = -2.34 * dt_range + 791.5  # Calculate corresponding y values
    # plt.plot(dt_range, tf_line, 'r--', label='y = -2.34x + 791.5')

    # # Calculate y-axis limits based on data points
    # y_min = min(tf_values) - 10  # Add some padding
    # y_max = max(tf_values) + 10
    # plt.ylim(y_min, y_max)  # Set y-axis limits

    # plt.show()

    #! Second plot - Scatter plot with inset temperature plots
    #fig, ax = plt.subplots(figsize=(12, 10))

    # Lists to store values
    tf_values = []
    dt_values = []
    colors = []

    # Extract values from file_ids
    for file_id in stress_data:
        tf_num = int(file_id.split('TF_')[0])
        dt_num = int(file_id.split('TD_')[0].split('_')[1])
        
        tf_values.append(tf_num)
        dt_values.append(dt_num)
        
        temps = NB_temps[file_id]
        if 0 in temps:
            colors.append('gray')
        elif temps[-1] < temps[0]:
            colors.append('red')
        elif temps[-1] > temps[0]:
            colors.append('green')

    # Calculate line parameters
    dt_range = np.array([min(dt_values), max(dt_values)])  # Create x-range for line
    tf_line = -2.34 * dt_range + 791.5  # Calculate corresponding y values

    # Calculate y-axis limits based on data points
    y_min = min(tf_values) - 10  # Add some padding
    y_max = max(tf_values) + 10

    # Create scatter plot with inset stress evolution plots
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create main scatter plot
    scatter = ax.scatter(dt_values, tf_values, c=colors, s=100)

    # Now add inset plots for each point
    for i, (dt, tf, file_id) in enumerate(zip(dt_values, tf_values, stress_data.keys())):
        # Create small inset axes for this point
        inset_ax = inset_axes(ax,
                            width=0.7,  # reduced from 1.0
                            height=0.5,  # reduced from 0.7
                            loc='center',
                            bbox_to_anchor=(dt, tf, 0.0, 0.0),
                            bbox_transform=ax.transData)
        
        # Plot temperature data in inset
        temps_array = np.array(NB_temps[file_id])
        inset_ax.plot(range(2, days+1), temps_array-273.15, 'k-', linewidth=0.5)
        inset_ax.set_xticks([])  # Remove x ticks for clarity
        inset_ax.set_yticks([])  # Remove y ticks for clarity

    # Main plot formatting
    ax.set_xlabel('DT (Temperature Difference [°C])')
    ax.set_ylabel('TF (Fluid Temperature [°C])')
    ax.set_title('Norton-Bailey Representative Temperature Behavior')


    # Add dashed black line to second plot
    ax.plot(dt_range, tf_line, 'k--', label='y = -2.34x + 791.5', zorder=10, linewidth=2)
    ax.set_ylim(y_min, y_max)  # Set y-axis limits

    # plt.show()
    plt.close()

    # #! Third plot - Temperature differences as text
    # fig, ax = plt.subplots(figsize=(12, 10))

    # # Add dashed red line
    # ax.plot(dt_range, tf_line, 'r--', label='y = -41x + 330.28')

    # # Add text annotations for temperature differences
    # for dt, tf, file_id in zip(dt_values, tf_values, stress_data.keys()):
    #     temps = NB_temps[file_id]
    #     temp_diff = temps[-1] - temps[0]  # Last temp minus first temp
        
    #     # Format the temperature difference
    #     text_color = 'gray'
    #     if 0 not in temps:  # Only change color if no zeros present
    #         text_color = 'red' if temp_diff < 0 else 'green'
        
    #     # Round to 2 decimal places and add text annotation
    #     ax.text(dt, tf, f'{temp_diff:.2f}', 
    #             color=text_color,
    #             ha='center', va='center')  # Center the text on the point

    # # Main plot formatting
    # ax.set_xlabel('DT (Temperature Difference [°C])')
    # ax.set_ylabel('TF (TF (Fluid Temperature [°C])')
    # ax.set_title('Norton-Bailey Representative Temperature Differences: End-First Day')
    # ax.grid(True)

    # # Legend
    # legend_elements = [
    #     plt.Line2D([0], [0], color='gray', marker='$0$', markersize=10, linestyle='None', label='Contains Zero Temperature'),
    #     plt.Line2D([0], [0], color='red', marker='$-$', markersize=10, linestyle='None', label='Temperature Decreased'),
    #     plt.Line2D([0], [0], color='green', marker='$+$', markersize=10, linestyle='None', label='Temperature Increased'),
    #     plt.Line2D([0], [0], color='r', linestyle='--', label='y = -41x + 330.28')
    # ]
    # ax.legend(handles=legend_elements)

    # ax.set_ylim(y_min, y_max)  # Set y-axis limits

    # plt.show()

    # #! Final NB temp divided by fluid temp
    # fig, ax = plt.subplots(figsize=(12, 10))

    # # Add dashed red line
    # ax.plot(dt_range, tf_line, 'r--', label='y = -41x + 330.28')

    # # Add text annotations for simple ratio calculations
    # for dt, tf, file_id in zip(dt_values, tf_values, stress_data.keys()):
    #     temps = NB_temps[file_id]
    #     last_temp = temps[-1]  # Keep in Kelvin
        
    #     # Calculate simple ratio x/tf
    #     try:
    #         ratio = last_temp / (tf+273.15)
    #         # Format the ratio
    #         text_color = 'gray'
    #         if 0 not in temps:  # Only change color if no zeros present
    #             text_color = 'red' if ratio < 1 else 'green'
            
    #         # Round to 2 decimal places and add text annotation
    #         ax.text(dt, tf, f'{ratio:.2f}', 
    #                 color=text_color,
    #                 ha='center', va='center')  # Center the text on the point
    #     except ZeroDivisionError:
    #         ax.text(dt, tf, 'div/0', 
    #                 color='gray',
    #                 ha='center', va='center')

    # # Main plot formatting
    # ax.set_xlabel('DT (Temperature Difference [°C])')
    # ax.set_ylabel('TF (TF (Fluid Temperature [°C])')
    # ax.set_title('Final NB Representative Temperature divided by Fluid Temperature [x/TF]')
    # ax.grid(True)
    # ax.set_ylim(y_min, y_max)  # Use same y-axis limits as other plots

    # # Legend
    # legend_elements = [
    #     plt.Line2D([0], [0], color='gray', marker='$0$', markersize=10, linestyle='None', label='Contains Zero/Invalid'),
    #     plt.Line2D([0], [0], color='red', marker='$-$', markersize=10, linestyle='None', label='Ratio < 1'),
    #     plt.Line2D([0], [0], color='green', marker='$+$', markersize=10, linestyle='None', label='Ratio ≥ 1'),
    #     plt.Line2D([0], [0], color='r', linestyle='--', label='y = -41x + 330.28')
    # ]
    # ax.legend(handles=legend_elements)

    # plt.show()

    # #! T/DT ratio
    # fig, ax = plt.subplots(figsize=(12, 10))

    # # Add dashed red line
    # ax.plot(dt_range, tf_line, 'r--', label='y = -41x + 330.28')

    # # Add text annotations for T/DT ratio calculations
    # for dt, tf, file_id in zip(dt_values, tf_values, stress_data.keys()):
    #     temps = NB_temps[file_id]
    #     last_temp = temps[-1]  # Keep in Kelvin
        
    #     # Calculate T/DT ratio
    #     try:
    #         ratio = last_temp / (dt+273.15)
    #         # Format the ratio
    #         text_color = 'gray'
    #         if 0 not in temps:  # Only change color if no zeros present
    #             text_color = 'red' if ratio < 1 else 'green'
            
    #         # Round to 2 decimal places and add text annotation
    #         ax.text(dt, tf, f'{ratio:.2f}', 
    #                 color=text_color,
    #                 ha='center', va='center')  # Center the text on the point
    #     except ZeroDivisionError:
    #         ax.text(dt, tf, 'div/0', 
    #                 color='gray',
    #                 ha='center', va='center')

    # # Main plot formatting
    # ax.set_xlabel('DT (Temperature Difference [°C])')
    # ax.set_ylabel('TF (TF (Fluid Temperature [°C])')
    # ax.set_title('Final NB Representative Temperature divided by Temperature Difference [x/DT]')
    # ax.grid(True)
    # ax.set_ylim(y_min, y_max)  # Use same y-axis limits as other plots

    # # Legend
    # legend_elements = [
    #     plt.Line2D([0], [0], color='gray', marker='$0$', markersize=10, linestyle='None', label='Contains Zero/Invalid'),
    #     plt.Line2D([0], [0], color='red', marker='$-$', markersize=10, linestyle='None', label='Ratio < 1'),
    #     plt.Line2D([0], [0], color='green', marker='$+$', markersize=10, linestyle='None', label='Ratio ≥ 1'),
    #     plt.Line2D([0], [0], color='r', linestyle='--', label='y = -41x + 330.28')
    # ]
    # ax.legend(handles=legend_elements)

    # plt.show()

    # #! Scatter plot with inset stress plots
    # fig, ax = plt.subplots(figsize=(12, 10))

    # # Create main scatter plot
    # scatter = ax.scatter(dt_values, tf_values, c=colors, s=100)

    # # Now add inset plots for each point
    # for i, (dt, tf, file_id) in enumerate(zip(dt_values, tf_values, stress_data.keys())):
    #     # Determine border color based on temperature behavior
    #     temps = NB_temps[file_id]
    #     if 0 in temps:
    #         border_color = 'black'
    #     elif temps[-1] < temps[0]:
    #         border_color = 'red'
    #     elif temps[-1] > temps[0]:
    #         border_color = '#32CD32'  # Lime green color
        
    #     # Create small inset axes for this point
    #     inset_ax = inset_axes(ax,
    #                          width=0.7,  # reduced from 1.0
    #                          height=0.5,  # reduced from 0.7
    #                          loc='center',
    #                          bbox_to_anchor=(dt, tf, 0.0, 0.0),
    #                          bbox_transform=ax.transData)
        
    #     # Set the border color and width after creation
    #     for spine in inset_ax.spines.values():
    #         spine.set_color(border_color)
    #         spine.set_linewidth(2)
        
    #     # Plot stress data in inset
    #     df = stress_data[file_id]
    #     inset_ax.plot(df['vmRoutsAll'], 'k-', linewidth=0.5)
    #     inset_ax.set_xticks([])
    #     inset_ax.set_yticks([])

    # # Main plot formatting
    # ax.set_xlabel('DT (Temperature Difference [°C])')
    # ax.set_ylabel('TF (Fluid Temperature [°C])')
    # ax.set_title(' Stress Max for each Case')
    # ax.grid(True)

    # # Legend
    # legend_elements = [
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Invalid'),
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Temperature Decreased'),
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#32CD32', markersize=10, label='Temperature Increased')
    # ]
    # ax.legend(handles=legend_elements)

    # # Add dashed red line
    # ax.plot(dt_range, tf_line, 'r--', label='y = -41x + 330.28')
    # ax.set_ylim(y_min, y_max)  # Set y-axis limits

    # plt.show()


    # #! Plot with x/(tf+dt) ratio calculation as text
    # fig, ax = plt.subplots(figsize=(12, 10))

    # # Add dashed red line
    # ax.plot(dt_range, tf_line, 'r--', label='y = -41x + 330.28')

    # # Add text annotations for ratio calculations
    # for dt, tf, file_id in zip(dt_values, tf_values, stress_data.keys()):
    #     temps = NB_temps[file_id]
    #     last_temp = temps[-1]  # Keep in Kelvin
        
    #     # Calculate x/(tf+dt) ratio
    #     try:
    #         ratio = last_temp / ((tf+273.15) + (dt+273.15))
    #         # Format the ratio
    #         text_color = 'gray'
    #         if 0 not in temps:  # Only change color if no zeros present
    #             text_color = 'red' if ratio < 1 else 'green'
            
    #         # Round to 2 decimal places and add text annotation
    #         ax.text(dt, tf, f'{ratio:.2f}', 
    #                 color=text_color,
    #                 ha='center', va='center')  # Center the text on the point
    #     except ZeroDivisionError:
    #         ax.text(dt, tf, 'div/0', 
    #                 color='gray',
    #                 ha='center', va='center')

    # # Main plot formatting
    # ax.set_xlabel('DT (Temperature Difference [°C])')
    # ax.set_ylabel('TF (TF (Fluid Temperature [°C])')
    # ax.set_title('Final NB Representative Temperature divided by Max Temperature [x/(TF+DT)]')
    # ax.grid(True)
    # ax.set_ylim(y_min, y_max)  # Use same y-axis limits as other plots

    # # Legend
    # legend_elements = [
    #     plt.Line2D([0], [0], color='gray', marker='$0$', markersize=10, linestyle='None', label='Contains Zero/Invalid'),
    #     plt.Line2D([0], [0], color='red', marker='$-$', markersize=10, linestyle='None', label='Ratio < 1'),
    #     plt.Line2D([0], [0], color='green', marker='$+$', markersize=10, linestyle='None', label='Ratio ≥ 1'),
    #     plt.Line2D([0], [0], color='r', linestyle='--', label='y = -41x + 330.28')
    # ]
    # ax.legend(handles=legend_elements)

    # plt.show()


#!------------------------------------------------------------------------------------------------
#! Calculating damage using FEA files
daily_damage_FEA = {}
predicted_daily_damage_NB = {file_id: {day: np.zeros(days) for day in range(days-1)} for file_id in stress_data.keys()}
times_of_max_stress = {}
max_stress_FEA = {}
max_stresses_nb = {file_id: {day: np.zeros(days) for day in range(days-1)} for file_id in stress_data.keys()}
file_validity = {}
total_damage_NB = {file_id: {day: 0 for day in range(days-1)} for file_id in stress_data.keys()}
total_damage_FEA = {file_id: np.nan for file_id in stress_data.keys()}
total_damage_NB_adjusted = {file_id: {day: 0 for day in range(days-1)} for file_id in stress_data.keys()}
lifetime_NB = {file_id: {day: 0 for day in range(days-1)} for file_id in stress_data.keys()}
lifetime_FEA = {file_id: 0 for file_id in stress_data.keys()}
accuracy = {file_id: {day: 0 for day in range(days-1)} for file_id in stress_data.keys()}
scale_factor = {file_id: {day: 0 for day in range(days-1)} for file_id in stress_data.keys()}
temperature_profile = {}

#for instantanous damage calculations
stresses = {file_id: 0 for file_id in stress_data.keys()}
temps = {file_id: 0 for file_id in stress_data.keys()}
time_damages = {file_id: 0 for file_id in stress_data.keys()}

count = 0
for file_id in stress_data.keys():

    #! FOR DEBUGGING ONLY trying to view results for a specific file
    # if file_id != "300TF_150TD_20SUB_14PER":
    #     count += 1
    #     print("file not correct ", count)
    #     continue
    # else:
    #     print("file correct")

    # obtaining stress data
    sigmas = stress_data[file_id]['vmRoutsAll']
   
    # Find positions of identifiers
    tf_pos = file_id.find('TF')
    td_pos = file_id.find('TD')
    
    # Get digits before each identifier
    tf_num = int(file_id[tf_pos-3:tf_pos])
    td_num = int(file_id[td_pos-3:td_pos])
    daily_steps = int(period_num[file_id]*substep_num[file_id])
    days = 20  #! hardcoded
    
    # # Used for debugging (only x files at a time)
    # if count >100:
    #     break

    #! Calculating FEA damage
    times = np.linspace(0,days*period_num[file_id],days*period_num[file_id]*substep_num[file_id]+1)
    t_op = 12
    t_ph = 0.5
    Ts = bc.buildCrownTemp_wPH(times,period_num[file_id],days,substep_num[file_id],t_op,t_ph,Tf=tf_num,dT=td_num)
    temperature_profile[file_id] = Ts
    # plt.figure()
    # plt.plot(times, Ts, 'o', linestyle = 'none')
    # plt.show()
    daily_damage_FEA[file_id] = lc.calcCreep_laporte(sigmas,Ts,times,period_num[file_id],days)
    total_damage_FEA[file_id] = sum(daily_damage_FEA[file_id])
    #print("FEA damage for 20 days ", file_id, " is ", total_damage_FEA[file_id])
    lifetime_FEA[file_id] = (1-total_damage_FEA[file_id])/daily_damage_FEA[file_id][-1] #SOH/(last day damage)

    #------ begin instantaneous damage analysis
    # we will analyze the damage as a function of temperature and stress
    stresses_mpa, temps_c, time_dmg = lc.instantaneous_dmg(sigmas,Ts,times)
    stresses[file_id] = stresses_mpa[1:]
    temps[file_id] = temps_c[1:]
    time_damages[file_id] = time_dmg

    # Debugging Code:
    # plotting damage as a function of temperature and stress for INDIVIDUAL files (only uncomment if you want to see specific file)
    # plt.figure()
    # # Create color mapping
    # dmg_min = min(time_damages[file_id])
    # dmg_max = max(time_damages[file_id])
    # colors = plt.cm.Reds((np.array(time_damages[file_id]) - dmg_min) / (dmg_max - dmg_min))

    # # Create scatter plot and store the scatter object
    # scatter = plt.scatter(temps[file_id], stresses[file_id], c=time_damages[file_id], cmap='Reds', s=10)

    # # Add labels and title
    # plt.xlabel('Temperature used in Dmg Equation [C]')
    # plt.ylabel('Stress used in Dmg Equation [MPa]')
    # plt.title('Damage as a function of temperature and stress')

    # # Add colorbar using the scatter object
    # plt.colorbar(scatter, label='Damage Rate')

    # plt.grid(True, alpha=1)
    # plt.show()
    #----- end instantaneous damage analysis



    # storing FEA peak stresses to be used later

    ### vvv frankie's og way of finding maxes for each file. Sometimes returns incorrect maxes if res. stresses are extreme
    daily_max_stresses = np.zeros(days)
    max_stress_times = []
    for day in range(days): # finding max stresses
        start_idx = day * daily_steps
        end_idx = (day + 1) * daily_steps
        
        # Find max stress and its index within this day
        index_of_max_stress = np.argmax(sigmas[start_idx:end_idx])
        daily_max_stresses[day] = sigmas[start_idx:end_idx][(index_of_max_stress+start_idx)]
        period_num[file_id]*substep_num[file_id]
        # Store the actual time when this maximum occurs
        max_stress_times.append((index_of_max_stress+start_idx)/(substep_num[file_id]))  # dividing by substep_num to get the time in hours
    times_of_max_stress[file_id] = max_stress_times
    max_stress_FEA[file_id] = daily_max_stresses
    ### ^^^ frankie's og way of finding maxes for each file

    ### new max finding method implemented by jwenner, finds operational maxes and avoids residual maxes that are higher than operational maxes
    # only runs if an extreme residual is detected
    # check if the max location of last day is during operation or not
    ind_last    =sigmas.size
    t_off       =period_num[file_id] - t_op - 2*t_ph
    ind_last_op =ind_last - (t_off+t_ph+2)*substep_num[file_id] # i'm assuming the last possible time an operational max could occur is offtime + preheat + buffer before end
    if max_stress_times[-1]*substep_num[file_id] > ind_last_op:
    
        daily_max_stress_inds        =processingFuns.getPeakOpStress(sigmas, period_num[file_id], substep_num[file_id], period_num[file_id] - t_op - 2*t_ph)
        times_of_max_stress[file_id] =daily_max_stress_inds/substep_num[file_id]
        max_stress_FEA[file_id]      =sigmas[daily_max_stress_inds].values
    ###

    #now, all FEA data is stored and ready for plotting 
    if tf_num == 525 and td_num == 140:
        fontsize=14
        stressFig, stressAx = plt.subplots()
        stressAx.plot(np.linspace(0,sigmas.size/(substep_num[file_id]*period_num[file_id]),int(substep_num[file_id]*period_num[file_id]*days)+1), sigmas)
        stressAx.scatter(max_stress_times[0]/period_num[file_id], daily_max_stresses[0], color='r', label='initial' )
        stressAx.scatter(max_stress_times[-1]/period_num[file_id], daily_max_stresses[-1], color='k', label='final')
        stressAx.set_xlabel('cycle number',fontsize=fontsize)
        stressAx.set_ylabel('FEA stress (MPa)', fontsize=fontsize)
        stressAx.tick_params(labelsize=fontsize)
        stressAx.legend(fontsize=fontsize)
        stressFig.savefig('imgs/stress_plot.png', dpi=300)
        plt.show()

    # before moving on to norton bailey, remove all data with invalid temperatures
    if any(temp <= 0 for temp in NB_temps[file_id]):    # if any temperature is less than or equal to 0, the data is invalid
        predicted_daily_damage_NB[file_id] = "data invalid"
        file_validity[file_id] = False
        #print("Data invalid for file_id: ", file_id, "---- skipping file")
        continue
    else:
        file_validity[file_id] = True

    #! Calculating NB predicted damage (assumes "rectangular" stress profile)
    # simulate the stress profile using each days NB rep. temp.
    sigma_eq = max_stress_FEA[file_id][0]   # the first day's max stress
    # loop through each day
    for day in range(days-1):   #! RECALL THAT NB BY DEFINITION STARTS ON DAY 2
        simulation_temp = NB_temps[file_id][day]    # NB_temps[file_id][day] returns 19 temperatures: 1 for each day
        youngs_modulus = E_temps[file_id][day] # using the youngs modulus at the current day
        for simulated_day in range(days):   # this loop simulates the 20 day profile based on the current day's NB rep. temp. (simulation_temp)
            #print("simulating day ", simulated_day+1, " based on NB rep. temp. from day ", (day+2))   # day+2 because NB starts at day 2
            if simulated_day == 0:
                max_stresses_nb[file_id][day][simulated_day] = sigma_eq # the first day is the peak stress
            else:
                sigma_r = nb.norton_bailey_gonzalez(sigma_eq*10**6,(simulated_day+1)*60*60,youngs_modulus,simulation_temp) # calculating the relaxed stress up to the simulated day
                #print("Relaxed stress at day ", simulated_day, " is ", sigma_r)
                max_stresses_nb[file_id][day][simulated_day] = -sigma_r/(10**6)+sigma_eq # MPA - calculating the max stress at the simulated day
            #print("max stress at day ", simulated_day, " is ", max_stresses_nb[file_id][day][simulated_day])

    #! Damage calculations
    # now that nb has predicted stress for each day, we can calculate the damage for each day      
    for day in range(days-1): # RECALL THAT NB BY DEFINITION STARTS ON DAY 2
        # Create array of stresses for NB prediction
        daily_steps = substep_num[file_id] * period_num[file_id]
        base_array = np.array([])
        
        # This is essentially creating a 1 hour rectangularstress profile for each day
        small_value = 0.00000001  # Define small value to use instead of zero
        for i in range(days):
            # Create daily pattern: small values for first half instead of zeros
            daily_zeros_first = np.full(daily_steps // 2, small_value)
            # Stress values repeated for 60/substep_num steps
            stress_period = np.repeat(max_stresses_nb[file_id][day][i], int(60/substep_num[file_id]))
            # Remaining small values to complete the day instead of zeros
            remaining_zeros = np.full(daily_steps - (daily_steps // 2) - int(60/substep_num[file_id]), small_value)
            # Combine the patterns for this day
            daily_pattern = np.concatenate([daily_zeros_first, stress_period, remaining_zeros])
            base_array = np.append(base_array, daily_pattern)
        # Prepend zero to the array
        sigmas_nb = np.concatenate(([0], base_array))
        # Assuring the stress profile is correct
        # plt.figure()
        # plt.plot(times, sigmas_nb)
        # plt.show()
        # now sigmas_nb is ready to be used in the damage calculation
        #!!!!! what temp should be used?
        crown_temp = tf_num+td_num
        predicted_daily_damage_NB[file_id][day] = lc.calcCreep_laporte(sigmas_nb,crown_temp,times,period_num[file_id],days)   # should return a list of 20 values - 1 damage per each day
        total_damage_NB[file_id][day] = sum(predicted_daily_damage_NB[file_id][day])
        #print("NB damage based on day ", day+1," is ", total_damage_NB[file_id][day])
        #print("FEA damage should be ", total_damage_FEA[file_id])
        
        # Scale factor : equating stress profiles at the last day to be used as a scaling factor
        scale_factor[file_id][day] = daily_damage_FEA[file_id][day]/predicted_daily_damage_NB[file_id][day][day]
        #accuracy[file_id][day] = total_damage_NB[file_id][day]*scale_factor[file_id][day]/total_damage_FEA[file_id]
        total_damage_NB_adjusted[file_id][day] = total_damage_NB[file_id][day]*scale_factor[file_id][day]
        lifetime_NB[file_id][day] = (1-total_damage_NB_adjusted[file_id][day]) / (scale_factor[file_id][day]*predicted_daily_damage_NB[file_id][day][-1])  #SOH/(last day damage)



    #! The following code plots the INDIVIDUAL file_id FEA max stresses versus the NB 20 day predicted max stresses
    # plt.figure(figsize=(10, 6))
    
    # # Plot FEA data
    # plt.plot(times_of_max_stress[file_id], max_stress_FEA[file_id], 'k-', linewidth=2)
    # plt.plot(times_of_max_stress[file_id], max_stress_FEA[file_id], 'ko', markersize=5, label='FEA Max Stresses')
    
    # # Create colormap for NB predictions
    # reds = plt.cm.Reds(np.linspace(0.2, 1, 19))  # 19 colors from light to dark red
    
    # # Plot all NB predictions
    # for day in range(19):  # 0 to 18
    #     #if day == 18:  #uncomment to see all other predictions
    #         plt.plot(times_of_max_stress[file_id], max_stresses_nb[file_id][day], '-', 
    #             color=reds[day], linewidth=1, 
    #             label=f'NB Prediction from Day {day+1}')
    #         plt.plot(times_of_max_stress[file_id], max_stresses_nb[file_id][day], 'o', 
    #             color=reds[day], markersize=4)
    
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Stress (MPa)')
    # plt.title('FEA Max Stresses over 20 days for file: ' + file_id)
    
    # # Move legend outside of plot
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # # Adjust layout to prevent legend cutoff
    # plt.tight_layout()
    # plt.show()


#! Insert plots of FEA vs. NB final day predictions

if detailed_plots:
    # Create scatter plot with inset stress evolution plots
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create main scatter plot
    scatter = ax.scatter(dt_values, tf_values, c='black', s=100)

    # Now add inset plots for each point
    for i, (dt, tf, file_id) in enumerate(zip(dt_values, tf_values, stress_data.keys())):
        if file_validity[file_id] == False:
            continue
        
        # Create small inset axes for this point
        inset_ax = inset_axes(ax,
                            width=0.7,  # reduced from 1.0
                            height=0.5,  # reduced from 0.7
                            loc='center',
                            bbox_to_anchor=(dt, tf, 0.0, 0.0),
                            bbox_transform=ax.transData)
        
        # Plot FEA data in black
        inset_ax.plot(times_of_max_stress[file_id], max_stress_FEA[file_id], 'k-', linewidth=0.5)
        inset_ax.plot(times_of_max_stress[file_id], max_stress_FEA[file_id], 'ko', markersize=1)
        
        # Calculate max differences for both modes
        max_diff_20th = np.max(np.abs(np.array(max_stress_FEA[file_id]) - np.array(max_stresses_nb[file_id][18])))
        max_diff_3days = max([np.max(np.abs(np.array(max_stress_FEA[file_id]) - np.array(max_stresses_nb[file_id][j]))) for j in range(3)])

        #! must chose below
        plot_only_20th_day = False
        plot_first_3_days = True
        # Plot final NB prediction in red
        if plot_only_20th_day:
            inset_ax.plot(times_of_max_stress[file_id], max_stresses_nb[file_id][18], 'r-', linewidth=0.5)
            inset_ax.plot(times_of_max_stress[file_id], max_stresses_nb[file_id][18], 'ro', markersize=1)
            # Add max difference text to the inset
            legend_elements = [
                plt.Line2D([], [], color='none', label=f'{max_diff_20th:.2f}')]
            inset_ax.legend(handles=legend_elements, loc='best', frameon=False, 
                        handlelength=0, handletextpad=0, fontsize=6)
        

        # OR : plot first 3 days of NB predictions

        if plot_first_3_days:
            # Create color gradient from light to dark red
            reds = [(1, 0.6, 0.6), (1, 0.3, 0.3), (1, 0, 0)]  # Light to dark red
            for i in range(3):
                inset_ax.plot(times_of_max_stress[file_id], max_stresses_nb[file_id][i], 
                            '-', color=reds[i], linewidth=0.5)
                inset_ax.plot(times_of_max_stress[file_id], max_stresses_nb[file_id][i], 
                            'o', color=reds[i], markersize=1)
                    # Add max difference text to the inset
            legend_elements = [
                plt.Line2D([], [], color='none', label=f'{max_diff_3days:.2f}')]
            inset_ax.legend(handles=legend_elements, loc='best', frameon=False, 
                        handlelength=0, handletextpad=0, fontsize=6)
        
        # Remove all ticks and labels
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

    # Main plot formatting
    ax.set_xlabel('DT (Temperature Difference [°C])')
    ax.set_ylabel('TF (Fluid Temperature [°C])')
    ax.set_title('FEA Max Crown Stress vs. 20th Day Norton Bailey Prediction')
    #ax.grid(True)

    # Add dashed blue line and set y-axis limits
    ax.plot(dt_range, tf_line, 'b--', label='y = -2.34x + 791.5', zorder=10, linewidth=2)
    ax.set_ylim(y_min, y_max)

    # Add legend for main plot
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='black', label='Invalid Case'),
        plt.Line2D([0], [0], color='b', linestyle='--', label='y = -2.34x + 791.5')
    ]
    ax.legend(handles=legend_elements)

    plt.show()

    #! Lifetime percent error plot
    lifetime_error = {}
    # Create scatter plot with inset lifetime ratio plots
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create main scatter plot
    scatter = ax.scatter(dt_values, tf_values, c='black', s=100)

    # Now add inset plots for each point
    for i, (dt, tf, file_id) in enumerate(zip(dt_values, tf_values, stress_data.keys())):
        if file_validity[file_id] == False:
            continue
        
        # Create small inset axes for this point
        inset_ax = inset_axes(ax,
                            width=0.7,  # reduced from 1.0
                            height=0.5,  # reduced from 0.7
                            loc='center',
                            bbox_to_anchor=(dt, tf, 0.0, 0.0),
                            bbox_transform=ax.transData)
        
        # Create x-axis values (days 2-20)
        days_array = np.arange(2, 21)  # NB predictions start from day 2
        
        # Get lifetime error values for this file_id
        lifetime_error[file_id] = [(lifetime_NB[file_id][day]-lifetime_FEA[file_id])/lifetime_FEA[file_id] for day in range(len(days_array))]
        
        # Plot lifetime ratio data
        inset_ax.plot(days_array, lifetime_error[file_id], 'k-', linewidth=1)
        
        # Find min and max values
        min_val = np.min(lifetime_error[file_id])  
        max_val = np.max(lifetime_error[file_id]) 
        

        # Set y-axis limits slightly beyond min/max for padding
        y_padding = (max_val - min_val) * 0.1  # 10% padding
        inset_ax.set_ylim(min_val - y_padding, max_val + y_padding)
        
        # Add legend with min/max values
        legend_elements = [
            plt.Line2D([], [], color='none', label=f'{max_val:.2f}'),
            plt.Line2D([], [], color='none', label=f'{min_val:.2f}')
        ]
        inset_ax.legend(handles=legend_elements, loc='best', frameon=False, 
                    handlelength=0, handletextpad=0, fontsize=6)
        
        # Remove all ticks and labels
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

    # Main plot formatting
    ax.set_xlabel('DT (Temperature Difference [°C])')
    ax.set_ylabel('TF (Fluid Temperature [°C])')
    ax.set_title('Lifetime Percent Error')
    #ax.grid(True)


    # Add dashed blue line and set y-axis limits
    ax.plot(dt_range, tf_line, 'b--', label='y = -2.34x + 791.5', zorder=10, linewidth=2)
    ax.set_ylim(y_min, y_max)

    # Add legend for main plot
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='black', label='Invalid Case'),
        plt.Line2D([0], [0], color='b', linestyle='--', label='y = -2.34x + 791.5'),
    ]
    ax.legend(handles=legend_elements)

    plt.show()

    # #! Viewing lifetime error for specific file_id
    # file_id = "300TF_140TD_30SUB_14PER"
    # plt.figure(figsize=(10, 6))

    # # Create x-axis values (days 2-20)
    # days_array = np.arange(2, 21)  # NB predictions start from day 2

    # # Calculate lifetime errors
    # lifetime_errors = [abs((lifetime_NB[file_id][day]-lifetime_FEA[file_id])/lifetime_FEA[file_id]) for day in range(len(days_array))]
    # max_val = np.max(lifetime_errors)
    # min_val = np.min(lifetime_errors)

    # # Create the plot
    # plt.plot(days_array, lifetime_errors, 'k-', linewidth=2, label='Lifetime Error')
    # plt.plot(days_array, lifetime_errors, 'ko', markersize=6)

    # # Add labels and title
    # plt.xlabel('Day of NB Temperature')
    # plt.ylabel('Lifetime Error')
    # plt.xticks(days_array)
    # plt.title(f'Lifetime Error vs Day of NB Temperature for {file_id}')

    # # Add statistics box using annotate with automatic positioning
    # stats_text = f'Max Error: {max_val:.2f}\nMin Error: {min_val:.2f}'
    # plt.annotate(stats_text,
    #             xy=(0.02, 0.98),  # Position relative to axes (0,0 is bottom left, 1,1 is top right)
    #             xycoords='axes fraction',
    #             bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
    #             verticalalignment='top')

    # plt.show()

#! Contour plot of minimum fea days required to reach a certain lifetime error
#TODO CHANGE THIS TO TRUE TO RUN CONTOUR AND OTHER ERROR ANALYSIS PLOTS

if error_analysis:
    # Set error threshold (as a decimal)
    error_threshold = [0.20]
    for error in error_threshold:
        # Create regular grids for TF and DT
        tf_unique = sorted(list(set([int(file_id.split('TF_')[0]) for file_id in stress_data.keys()])))
        dt_unique = sorted(list(set([int(file_id.split('TD_')[0].split('_')[1]) for file_id in stress_data.keys()])))

        # Initialize array to store minimum days
        min_days_grid = np.full((len(tf_unique), len(dt_unique)), np.nan)

    # Fill the grid with minimum days needed
    for file_id in stress_data.keys():
        # Extract TF and DT from file_id
        tf = int(file_id.split('TF_')[0])
        dt = int(file_id.split('TD_')[0].split('_')[1])
        
        # Find indices in our grid
        tf_idx = tf_unique.index(tf)
        dt_idx = dt_unique.index(dt)
        
        if not file_validity[file_id]:
            min_days_grid[tf_idx, dt_idx] = 7  # Set invalid files directly to 7
            continue
        
        # Get absolute errors
        errors = np.abs(lifetime_error[file_id])
        
        # Initialize min_days to 7 (our cap)
        min_days = 7
        
        # Find the last sequence where error stays below threshold
        for day in range(len(errors)):
            if errors[day] <= error:
                # Check if error stays below threshold until the end
                if all(err <= error for err in errors[day:]):
                    min_days = day + 2  # Add 2 because NB starts at day 2
                    break
        
        min_days_grid[tf_idx, dt_idx] = min_days

    # Modify the min_days_grid to cap values at 7
    min_days_grid_capped = min_days_grid.copy()
    min_days_grid_capped[min_days_grid_capped >= 7] = 7

    # Create contour plot with interpolation and data points
    plt.figure(figsize=(12, 10))

    # Create initial meshgrid
    dt_mesh, tf_mesh = np.meshgrid(dt_unique, tf_unique)

    # Create levels from 2 to 7 with resolution of 1
    levels = np.arange(2, 8, 1)  # [2, 3, 4, 5, 6, 7]

    # Use scipy's interpolate to fill the gaps
    from scipy import interpolate

    # Create coarser meshgrid for more stable interpolation
    dt_fine = np.linspace(min(dt_unique), max(dt_unique), 30)
    tf_fine = np.linspace(300, 550, 30)  #! Explicitly set TF range HARDCODED
    dt_mesh_fine, tf_mesh_fine = np.meshgrid(dt_fine, tf_fine)

    # Get valid (non-NaN) points
    valid_points = ~np.isnan(min_days_grid_capped)
    points = np.column_stack((dt_mesh[valid_points], tf_mesh[valid_points]))
    values = min_days_grid_capped[valid_points]

    # Store original data points for scatter plot
    dt_data = dt_mesh[valid_points]
    tf_data = tf_mesh[valid_points]

    # Interpolate using only 'nearest' method
    filled_grid = interpolate.griddata(points, values, (dt_mesh_fine, tf_mesh_fine), method='nearest')

    # Create the contour plot with interpolated data
    contour = plt.contourf(dt_fine, tf_fine, filled_grid, levels=levels, cmap='viridis', extend='both')

    # Create custom colorbar with modified labels and rectangular ends
    cbar = plt.colorbar(contour, label='Minimum Simulation Days Required', 
                    ticks=levels,
                    drawedges=True,
                    spacing='uniform',
                    orientation='vertical',
                    format='%d',
                    extendfrac='auto',
                    extendrect=True)  # Make the extensions rectangular

    # Modify colorbar labels
    tick_labels = [str(int(tick)) if tick < 7 else '>=7' for tick in levels]
    cbar.set_ticklabels(tick_labels)

    # Add scatter points for actual data : keep off unless debugging
    #plt.scatter(dt_data, tf_data, c='red', s=50, marker='o', label='Data Points', zorder=5)

    # Add labels and title
    plt.xlabel('DT (Temperature Difference [°C])')
    plt.ylabel('TF (Fluid Temperature [°C])')
    plt.title(f'Minimum Simulation Days Required for {error*100}% Lifetime Error')

    # Add the line y = -2.34x + 791.5
    dt_range = np.array([min(dt_unique), max(dt_unique)])
    tf_line = -2.34 * dt_range + 791.5
    plt.plot(dt_range, tf_line, 'k--', label='y = -2.34x + 791.5')
    plt.legend()

    # Set y-axis limits explicitly
    plt.ylim(300, 550)

    plt.show()

    #! Second plot (numbers only)
    fig, ax = plt.subplots(figsize=(12, 10))  # Create figure and axis objects

    # Create empty plot with same scale
    plt.scatter(dt_data, tf_data, c='none', alpha=0)
    plt.xlabel('DT (Temperature Difference [°C])')
    plt.ylabel('TF (Fluid Temperature [°C])')
    plt.title(f'Simulation Days Required for {error*100}% Lifetime Error')
    #plt.grid(True)

    # Create colormap and normalize values for circle colors
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=2, vmax=7)

    # Add circles and numbers at each valid point
    for dt, tf in zip(dt_data, tf_data):
        # Find corresponding file_id
        file_id = next((fid for fid in stress_data.keys() 
                       if int(fid.split('TF_')[0]) == tf and 
                          int(fid.split('TD_')[0].split('_')[1]) == dt), None)
        
        if not file_validity[file_id]:
            # Invalid point - show black dot
            plt.scatter(dt, tf, s=100, c='black', alpha=1)
            continue
            
        # Get value for valid point
        val = min_days_grid[tf_unique.index(tf), dt_unique.index(dt)]
        
        # Calculate circle size (scales from 100 to 400 based on values 2 to 7)
        size = 150 + (min(val, 7) - 2) * 70  # Cap size at 7 but not the displayed number
        
        # Plot circle
        scatter = plt.scatter(dt, tf, s=size, 
                   c=[cmap(norm(min(val, 7)))],  # Cap color at 7
                   alpha=0.9)
        
        # Add number without outline - show actual value
        plt.text(dt, tf, f'{int(val)}', 
                ha='center', va='center',
                color='white',
                fontweight='bold', 
                fontsize=12)

    # Add the line y = -2.34x + 791.5
    dt_range = np.array([min(dt_unique), max(dt_unique)])
    tf_line = -2.34 * dt_range + 791.5
    plt.plot(dt_range, tf_line, 'r--', label='y = -2.34x + 791.5')
    
    # Add colorbar using the scatter plot as reference
    plt.colorbar(scatter, label='Minimum Simulation Days Required',
                ticks=range(2, 8)).set_ticklabels([str(i) if i < 7 else '>=7' for i in range(2, 8)])

    # Create proper legend handles
    legend_elements = [
        plt.Line2D([0], [0], linestyle='--', color='r', label='y = -2.34x + 791.5'),
        plt.Line2D([0], [0], marker='o', color='black', label='Invalid Case', linestyle='None')
    ]
    plt.legend(handles=legend_elements)
    plt.show()

    #! Plot showing percent error at specific day
    # TODO change day to the day you want to analyze
    day_to_analyze = 6  # Easily changeable variable (day 2-20)
    day_index = day_to_analyze - 2  # Convert to index (NB starts at day 2)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create empty plot with same scale
    plt.scatter(dt_data, tf_data, c='none', alpha=0)
    plt.xlabel('DT (Temperature Difference [°C])')
    plt.ylabel('TF (Fluid Temperature [°C])')
    plt.title(f'Lifetime Error at Day {day_to_analyze}')

    # Create custom colormap from tab20c
    tab20c = plt.cm.tab20c
    colors = [
        tab20c(8),    # darkest green (0-5%)
        tab20c(9),    # medium green (5-10%)
        tab20c(10),    # lightest green (10-15%)
        tab20c(7),    # lightest red (15-20%)
        tab20c(6),    # medium red (20-25%)
        tab20c(5),    # darker red (25-30%)
        tab20c(4),    # very dark red (30-35%)
    ]
    custom_cmap = mcolors.ListedColormap(colors)

    # Define the boundaries for each category
    bounds = [0, 5, 10, 15, 20, 25, 30, 100]  # Upper bound set high to catch all values above 35
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

    # Add circles with categorized colors
    for dt, tf in zip(dt_data, tf_data):
        file_id = next((fid for fid in stress_data.keys() 
                    if int(fid.split('TF_')[0]) == tf and 
                        int(fid.split('TD_')[0].split('_')[1]) == dt), None)
        
        if not file_validity[file_id]:
            # Invalid point - show black dot
            plt.scatter(dt, tf, s=100, c='black', alpha=1)
            continue
            
        error = abs(lifetime_error[file_id][day_index]) * 100  # Convert to percentage
        
        # Plot circle with color based on error category
        scatter = plt.scatter(dt, tf, s=400, 
                            c=[error], 
                            cmap=custom_cmap,
                            norm=norm,
                            alpha=0.9)
        
        # Add error percentage text
        plt.text(dt, tf, f'{int(error)}%', 
                ha='center', va='center',
                color='white',
                fontweight='bold', 
                fontsize=8)

    # Add the line y = -2.34x + 791.5
    dt_range = np.array([min(dt_unique), max(dt_unique)])
    tf_line = -2.34 * dt_range + 791.5
    plt.plot(dt_range, tf_line, 'r--', label='y = -2.34x + 791.5')

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tab20c(8), label='0-5%', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tab20c(9), label='5-10%', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tab20c(10), label='10-15%', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tab20c(7), label='15-20%', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tab20c(6), label='20-25%', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tab20c(5), label='25-30%', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tab20c(4), label='30+%', markersize=10),
        plt.Line2D([0], [0], marker='o', color='black', label='Invalid Case', markersize=10),
        plt.Line2D([0], [0], linestyle='--', color='r', label='y = -2.34x + 791.5')
    ]
    plt.legend(handles=legend_elements, 
            title='Error Ranges',
            bbox_to_anchor=(1.05, 1),  # Position legend outside plot
            loc='upper left')  # Align to upper left of bbox

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()  # Adjust plot to make room for legend

    plt.show()

#! TF DT plot showing last day damage rate vs total temperature; used to assess damage as a function of total temperature
# plt.figure(figsize=(12, 8))

# # Calculate total temperatures and prepare data
# damage_rates = []
# dt_values_for_color = []

# for file_id in stress_data.keys():
#     #if not file_validity[file_id]:
#     #    continue
        
#     # Extract TF and DT
#     tf = int(file_id.split('TF_')[0])
#     dt = int(file_id.split('TD_')[0].split('_')[1])

#     # Get last day's damage rate
#     last_day_damage = daily_damage_FEA[file_id][-1]
    
#     damage_rates.append(last_day_damage)
#     dt_values_for_color.append(dt)

# # Create color mapping
# dt_min = min(dt_values_for_color)
# dt_max = max(dt_values_for_color)
# colors = plt.cm.Reds((np.array(dt_values_for_color) - dt_min) / (dt_max - dt_min))

# # Create scatter plot and store the scatter object
# scatter = plt.scatter(file_total_temp.values(), damage_rates, c=dt_values_for_color, cmap='Reds', s=100)

# # Add labels and title
# plt.xlabel('Max Temperature (TF + DT) [°C]')
# plt.ylabel('Damage per Day')
# plt.title('Daily Damage Rate vs Max Temperature')

# # Add colorbar using the scatter object
# plt.colorbar(scatter, label='DT Value [°C]')

# plt.grid(True, alpha=0.3)
# plt.show()


#! Plotting damage as a function of both temperature and stress
#!!!!!! this takes a lot of power to produce, fix this and consider coloring each f_id differently
# plt.figure()
# for file in stress_data.keys():
#     damage_rates_all = []
#     temperatures_all =[]
#     stresses_all = []

#     # Create color mapping
#     dmg_min = min(time_damages[file_id])
#     dmg_max = max(time_damages[file_id])
#     colors = plt.cm.Reds((np.array(time_damages[file_id]) - dmg_min) / (dmg_max - dmg_min))

#     # Create scatter plot and store the scatter object
#     scatter = plt.scatter(temps[file_id], stresses[file_id], c=time_damages[file_id], cmap='Reds', s=5)

# # Add labels and title
# plt.xlabel('Temperature used in Dmg Equation [C]')
# plt.ylabel('Stress used in Dmg Equation [MPa]')
# plt.title('Damage as a function of temperature and stress')

# # Add colorbar using the scatter object
# plt.colorbar(scatter, label='Damage Rate')

# plt.grid(True)
# plt.show()


#! Instantaneous damage as a function of crown temperature
# fig, ax = plt.subplots(figsize=(12, 8))

# # Get unique DT values and create a colormap
# dt_values = [int(file_id.split('TD_')[0].split('_')[1]) for file_id in stress_data.keys()]
# dt_min, dt_max = min(dt_values), max(dt_values)
# norm = plt.Normalize(dt_min, dt_max)
# cmap = plt.cm.Reds

# # Plot each file's data
# for file_id in stress_data.keys():
#     if file_validity[file_id]:  # Only plot valid files
#         dt = int(file_id.split('TD_')[0].split('_')[1])
#         color = cmap(norm(dt))
#         ax.plot(temps[file_id], time_damages[file_id], '.', color=color, markersize=2)

# ax.set_xlabel('Crown Temperature [°C]')
# ax.set_ylabel('Instantaneous Damage Rate')
# ax.set_title('Instantaneous Damage Rate vs Crown Temperature')
# ax.set_yscale('log')  # Use log scale for damage rate due to large variations
# ax.grid(True, alpha=0.3)

# # Add colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# plt.colorbar(sm, ax=ax, label='Temperature Difference (DT) [°C]')

# plt.tight_layout()  # Adjust layout to prevent colorbar cutoff

# plt.show()

# #! what percent of day max damage occured
# significant_time_dmg = {}
# significant_intervals = {}
# significant_time_dmg_at_max = {}
# for file_id in stress_data.keys():
        
#     # Get total damage
#     total_dmg = sum(time_damages[file_id])
#     day_damage = daily_damage_FEA[file_id][-1]
#     #print("sum_total_dmg - daily_damage_FEA = (should be 0)", total_dmg - day_damage)
#     significant_time_dmg_at_max[file_id] = total_dmg/(max(time_damages[file_id]))*substep_num[file_id] # will be in hours
    
#     # Create array of (damage, index) pairs and sort by damage in descending order
#     dmg_with_idx = [(dmg, idx) for idx, dmg in enumerate(time_damages[file_id])]
#     dmg_with_idx.sort(reverse=True)
    
#     # Find how many intervals it takes to reach 90% of damage
#     cumulative_dmg = 0
#     threshold = 0.90 * total_dmg
#     intervals_needed = 0
#     significant_indices = []
    
#     for dmg, idx in dmg_with_idx:
#         cumulative_dmg += dmg
#         intervals_needed += 1
#         significant_indices.append(idx)
#         if cumulative_dmg >= threshold:
#             break
    
#     # Calculate percentage of time that causes 90% of damage
#     total_intervals = len(time_damages[file_id])
#     frac_time = (intervals_needed / total_intervals)
#     per = int(file_id.split('_')[3].replace('PER', ''))
#     significant_time_dmg[file_id] = frac_time*per
#     significant_intervals[file_id] = significant_indices

# # Create scatter plot of results
# plt.figure(figsize=(12, 8))

# # Extract TF and DT values for plotting
# tf_values = [int(file_id.split('TF_')[0]) for file_id in significant_time_dmg.keys()]
# dt_values = [int(file_id.split('TD_')[0].split('_')[1]) for file_id in significant_time_dmg.keys()]
# sig_dmg_times = list(significant_time_dmg.values())
# sig_dmg_times_at_max = list(significant_time_dmg_at_max.values())

# # Create scatter plot with color based on percentage
# scatter = plt.scatter(dt_values, tf_values, c=sig_dmg_times, cmap='Reds', s=100)

# plt.xlabel('Temperature Difference (DT) [°C]')
# plt.ylabel('Fluid Temperature (TF) [°C]')
# plt.title('Time Contributing to 90% of Total Damage')
# plt.colorbar(scatter, label='Hours [hr]')
# plt.grid(True, alpha=0.3)

# # Add the line y = -2.34x + 791.5
# dt_range = np.array([min(dt_values), max(dt_values)])
# tf_line = -2.34 * dt_range + 791.5
# plt.plot(dt_range, tf_line, 'k--', label='y = -2.34x + 791.5')
# plt.legend()

# plt.tight_layout()
# plt.show()   

# # Create scatter plot with color based on percentage
# scatter = plt.scatter(dt_values, tf_values, c=sig_dmg_times_at_max, cmap='Reds', s=100)

# plt.xlabel('Temperature Difference (DT) [°C]')
# plt.ylabel('Fluid Temperature (TF) [°C]')
# plt.title('Time Contributing to 100% of Total Damage at Max Stress and Temperature')
# plt.colorbar(scatter, label='Hours [hr]')
# plt.grid(True, alpha=0.3)

# # Add the line y = -2.34x + 791.5
# dt_range = np.array([min(dt_values), max(dt_values)])
# tf_line = -2.34 * dt_range + 791.5
# plt.plot(dt_range, tf_line, 'k--', label='y = -2.34x + 791.5')
# plt.legend()

# plt.tight_layout()
# plt.show()   


#! Plotting the maximum daily instantaneous damage rate
plt.figure()

# Initialize lists to store plot data
tf_values = []
dt_values = []
instant_max_daily_dmg_values = []
max_instant_dmg_value = {}

# Loop through all file_ids
for file_id in stress_data.keys():
    # Extract TF, DT, time_damages, substep_num, and period_num
    tf_num = int(file_id.split('TF_')[0])
    dt_num = int(file_id.split('TD_')[0].split('_')[1])
    per = period_num[file_id]
    sub = substep_num[file_id]
    time_dmg = time_damages[file_id][-per*sub:]

    
    # Find the maximum time damage
    max_time_dmg = max(time_dmg)
    
    # Calculate substep_length_hrs and substep_length_s
    substep_length_hrs = 1 / substep_num[file_id]
    substep_length_s = substep_length_hrs*60*60

    
    # Calculate instant_max_daily_dmg
    instant_max_daily_dmg = max_time_dmg / substep_length_s
    max_instant_dmg_value[file_id] = instant_max_daily_dmg
    #print("max instantaneous damage rate: ", instant_max_daily_dmg)
    #print("substep length: ", substep_length_s)
    
    # Collect values for plotting
    tf_values.append(tf_num)
    dt_values.append(dt_num)
    instant_max_daily_dmg_values.append(instant_max_daily_dmg)

# Create scatter plot with logarithmic color scaling
scatter = plt.scatter(dt_values, tf_values, 
                     c=instant_max_daily_dmg_values, 
                     cmap='Reds',
                     s=100,
                     norm=matplotlib.colors.LogNorm())  # This creates logarithmic color scaling

# Add labels and title
plt.xlabel('Temperature Difference (DT) [°C]')
plt.ylabel('Fluid Temperature (TF) [°C]')
plt.title('Maximum Instantaneous Damage Rate [Dmg/s]')

# Add colorbar using the scatter object
cbar = plt.colorbar(scatter, label='Instantaneous Max Daily Damage Rate [Dmg/s]')

# Format the colorbar ticks for better readability
cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
cbar.ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation())

# Adjust colorbar formatting
cbar.ax.tick_params(labelsize=8)  # Keep font size at 8
cbar.ax.yaxis.get_offset_text().set_fontsize(8)  # Keep scientific notation size at 8
cbar.ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=7))  # Keep number of ticks
cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=7, subs=(0.2, 0.4, 0.6, 0.8)))  # Keep minor ticks

# Add padding between ticks and labels
cbar.ax.tick_params(pad=5)  # Keep padding between ticks and labels

plt.grid(True, alpha=0.3)
plt.show()


#! Plotting the area under the curve for each dt tf case
# function for determining the area under the curve for a given day
def clean_stress_last_day(file_id):
    sub = substep_num[file_id]
    per = period_num[file_id]
    #substep_length_s = 3600/(sub)
    day_length = sub*per
    stress_for_file = stress_data[file_id]['vmRoutsAll'].iloc[-day_length:].values


    # lets find only the positive stress values for the day
    max_stress_time = np.argmax(stress_for_file)
    difference =0
    previous_stress = max(stress_for_file)
    flag = 0
    right_flag_index = None
    left_flag_index = None
    #finding the minumum on the right side of the max stress
    right_side_stress = stress_for_file[max_stress_time:]
    for stress in right_side_stress:
        #print("stress: ", stress, "previous_stress: ", previous_stress)
        difference = stress - previous_stress
        if difference < 0:
            flag = 0
            previous_stress = stress
        if difference > 0:
            if flag == 0:   
                right_flag_index = np.where(stress_for_file == previous_stress)[0][0]
            flag +=1
            previous_stress = stress
        if flag == 5:
            #print("Stress increasing again starting at index: ", right_flag_index)
            break
    
    #finding the minumum on the left side of the max stress
    left_side_stress = stress_for_file[0:max_stress_time]
    left_side_stress = left_side_stress[::-1]
    flag = 0
    previous_stress = max(stress_for_file)
    left_flag_index = None
    for stress in left_side_stress:
        difference = stress - previous_stress
        if difference < 0:
            flag = 0
            previous_stress = stress
        elif difference > 0:
            if flag == 0:
                left_flag_index = np.where(stress_for_file == previous_stress)[0][0]
            flag +=1
            previous_stress = stress
        if flag == 5:
            #print("Stress increasing again starting at index: ", left_flag_index)
            break
    
    # new stresses with cleaned data
    stress_for_file = stress_for_file[left_flag_index:right_flag_index]
    #plt.figure()
    #plt.title(f"Cleaned Stress for last day of file {file_id}")
    #plt.plot(stress_for_file, color = 'red')
    #plt.show()
    return stress_for_file

# calcualting the area under the curve for each dt tf case
area_under_curve = {}
for file_id in stress_data.keys():
    last_day_stress = clean_stress_last_day(file_id)
    area_under_curve[file_id] = np.average(last_day_stress) #np.trapz(last_day_stress, dx = 3600/substep_num[file_id])
    #calcualtes area as MPa * s

# making the plot

# --------------- Create scatter plot of area under the curve results ---------------
# plt.figure(figsize=(12, 8))

# # Extract TF and DT values for plotting
# tf_values = [int(file_id.split('TF_')[0]) for file_id in area_under_curve.keys()]
# dt_values = [int(file_id.split('TD_')[0].split('_')[1]) for file_id in area_under_curve.keys()]
# areas = list(area_under_curve.values())

# # Create scatter plot with color based on area under the curve
# scatter = plt.scatter(dt_values, tf_values, c=areas, cmap='Reds', s=100)

# # Add labels and title
# plt.xlabel('Temperature Difference (DT) [°C]')
# plt.ylabel('Fluid Temperature (TF) [°C]')
# plt.title('Area Under the Curve for Last Day Stress')

# # Add colorbar
# plt.colorbar(scatter, label='Area Under Curve [MPa·s]')

# # Add grid for better readability
# plt.grid(True, alpha=0.3)

# # Add the characteristic line y = -2.34x + 791.5
# dt_range = np.array([min(dt_values), max(dt_values)])
# tf_line = -2.34 * dt_range + 791.5
# plt.plot(dt_range, tf_line, 'k--', label='y = -2.34x + 791.5')
# plt.legend()

# plt.tight_layout()
# plt.show()

#! Making array of damages and areas
file_tuples = []
file_tuples2 = []
for file_id in stress_data.keys():
    dt = int(file_id.split('TD_')[0].split('_')[1])
    tf = int(file_id.split('TF_')[0])
    sub = substep_num[file_id]
    dmg = np.log(max_instant_dmg_value[file_id])
    area = area_under_curve[file_id]


    #making a tuple for this file
    file_tuples.append((dt, tf, dmg, area))
    file_tuples2.append((dt, tf, dmg, area))

    # adding data for dt<100
    if dt == 100 or (tf == 550 and dt==110):
        for new_dt in range(0,100,10):
            # retrieving data
            new_temps = temps[file_id][-14*sub:]
            dt_vals = new_temps-tf
            new_dmg = time_damages[file_id][-14*sub:]/(1/sub*60*60)

            # finding damage at new dt
            closest_idx = np.argmin(np.abs(np.array(dt_vals) - new_dt))
            new_dt_dmg = np.log(new_dmg[closest_idx])

            # adding to list
            file_tuples.append((new_dt, tf, new_dt_dmg, area))

    # populating second array for yielding cases
    if (dt == 140 and tf<425) or (dt==130 and tf>400 and tf<500) or (dt==120 and tf>475 and tf<525) or (dt==110 and tf>500):
        file_tuples2.append((dt, tf, dmg, area))
        # retrieving data
        new_temps = temps[file_id][-14*sub:]
        dt_vals = new_temps-tf
        new_dmg = time_damages[file_id][-14*sub:]/(1/sub*60*60)
        
        # used to debug
        # plt.figure()
        # plt.title(f"Damage and Temperature for {file_id}")
        # ax1 = plt.gca()
        # ax1.plot(new_temps, label='Temperature', color='red')
        # ax1.set_ylabel('Temperature [°C]', color='red')
        # ax1.tick_params(axis='y', labelcolor='red')
        # ax2 = ax1.twinx()
        # ax2.plot(new_dmg, label='Damage', color='blue')
        # ax2.set_ylabel('Damage Rate', color='blue')
        # ax2.tick_params(axis='y', labelcolor='blue')
        # lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        # plt.tight_layout()
        # plt.show()
        for new_dt in range(0,dt+10,10):
            # finding damage at new dt
            closest_idx = np.argmin(np.abs(np.array(dt_vals) - new_dt))
            new_dt_dmg = np.log(new_dmg[closest_idx])

            # adding to list
            file_tuples2.append((new_dt, tf, new_dt_dmg, area))


# First get all unique dt and tf values to determine array size
unique_dt = sorted(list(set(t[0] for t in file_tuples)))
unique_tf = sorted(list(set(t[1] for t in file_tuples)))

# Create empty arrays
damage_grid = np.zeros((len(unique_tf), len(unique_dt)))
damage_grid2 = np.zeros((len(unique_tf), len(unique_dt)))
area_grid = np.zeros((len(unique_tf), len(unique_dt)))
area_grid2 = np.zeros((len(unique_tf), len(unique_dt)))

# Fill the arrays
for dt, tf, dmg, area in file_tuples:
    dt_idx = unique_dt.index(dt)
    tf_idx = unique_tf.index(tf)
    damage_grid[tf_idx, dt_idx] = dmg
    area_grid[tf_idx, dt_idx] = area

# populating second array
for dt, tf, dmg, area in file_tuples2:
    dt_idx = unique_dt.index(dt)
    tf_idx = unique_tf.index(tf)
    damage_grid2[tf_idx, dt_idx] = dmg
    area_grid2[tf_idx, dt_idx] = area

# add hardcoded values to help interpolater
#! must hardcode missing points based on prior interpolation
    #"""
    # damage for non-yielding cases
    damage_grid[1, 21] = np.log(3.777e-10) # 325, 210
    damage_grid[4, 20] = np.log(1.257e-8) # 400, 200
    damage_grid[6, 15] = np.log(1.229e-8) # 450, 150
    damage_grid[7, 17] = np.log(2.734e-8) # 475, 170
    damage_grid[7, 20] = np.log(1.998e-8) # 475, 200
    damage_grid[8, 21] = np.log(3.462e-8) # 500, 210
    damage_grid[9, 12] = np.log(1.650e-8) # 525, 120
    damage_grid[9, 13] = np.log(1.818e-8) # 525, 130
    damage_grid[9, 15] = np.log(1.863e-8) # 525, 150
    damage_grid[9, 21] = np.log(4.599e-8) # 525, 210
    damage_grid[9, 22] = np.log(4.599e-8) # 525, 220
    damage_grid[10, 10] = np.log(1.508e-8) # 550, 100
    damage_grid[10, 13] = np.log(1.482e-8) # 550, 130
    damage_grid[10, 16] = np.log(1.257e-8) # 550, 160
    damage_grid[10, 17] = np.log(2.000e-8) # 550, 170
    damage_grid[10, 18] = np.log(3.381e-8) # 550, 180
    damage_grid[10, 20] = np.log(3.557e-8) # 550, 200
    damage_grid[10, 21] = np.log(3.557e-8) # 550, 210
    damage_grid[10, 22] = np.log(3.557e-8) # 550, 220
    #"""
    # damage for yielding cases
    damage_grid2[1, 21] = np.log(3.777e-10) # 325, 210
    damage_grid2[4, 20] = np.log(1.257e-8) # 400, 200
    damage_grid2[6, 15] = np.log(1.229e-8) # 450, 150
    damage_grid2[7, 17] = np.log(2.734e-8) # 475, 170
    damage_grid2[7, 20] = np.log(1.998e-8) # 475, 200
    damage_grid2[8, 21] = np.log(3.462e-8) # 500, 210
    damage_grid2[9, 12] = np.log(1.650e-8) # 525, 120
    damage_grid2[9, 13] = np.log(1.818e-8) # 525, 130
    damage_grid2[9, 15] = np.log(1.863e-8) # 525, 150
    damage_grid2[9, 21] = np.log(4.599e-8) # 525, 210
    damage_grid2[9, 22] = np.log(4.599e-8) # 525, 220
    damage_grid2[10, 10] = np.log(1.508e-8) # 550, 100
    damage_grid2[10, 13] = np.log(1.482e-8) # 550, 130
    damage_grid2[10, 16] = np.log(1.257e-8) # 550, 160
    damage_grid2[10, 17] = np.log(2.000e-8) # 550, 170
    damage_grid2[10, 18] = np.log(3.381e-8) # 550, 180
    damage_grid2[10, 20] = np.log(3.557e-8) # 550, 200
    damage_grid2[10, 21] = np.log(3.557e-8) # 550, 210
    damage_grid2[10, 22] = np.log(3.557e-8) # 550, 220


   #! must hardcode missing points based on prior interpolation
    #"""
    #areas for non-yielding cases
    area_grid[1, 21] = 170.401 # 325, 210

    area_grid[4,20] = 163.825 # 400, 200

    area_grid[6, 15] = 162.300# 450, 150

    area_grid[7, 17] = 137.761 # 475, 170

    area_grid[7, 20] = 109.121 # 475, 200

    area_grid[8, 21] = 192.226 # 500, 210

    area_grid[9, 12] =  131.517 # 525, 120

    area_grid[9, 13] = 118.922 # 525, 130

    area_grid[9, 15] = 102.597 # 525, 150

    area_grid[9, 21] = 182.729 # 525, 210

    area_grid[9, 22] = 182.729 # 525, 220

    area_grid[10, 10] = 118.303 # 550, 100

    area_grid[10, 13] = 102.807 # 550, 130

    area_grid[10, 16] = 96.836 # 550, 160

    area_grid[10, 17] = 110.92 # 550, 170

    area_grid[10, 18] = 152.035 # 550, 180

    area_grid[10, 20] = 203.65 # 550, 200

    area_grid[10, 21] = 203.65 # 550, 210

    area_grid[10, 22] = 203.65 # 550, 220
    #"""
    # areas for yielding cases
    area_grid2[1, 21] = 170.401 # 325, 210

    area_grid2[4,20] = 163.825 # 400, 200

    area_grid2[6, 15] = 162.300# 450, 150

    area_grid2[7, 17] = 137.761 # 475, 170

    area_grid2[7, 20] = 109.121 # 475, 200

    area_grid2[8, 21] = 192.226 # 500, 210

    area_grid2[9, 12] =  131.517 # 525, 120

    area_grid2[9, 13] = 118.922 # 525, 130

    area_grid2[9, 15] = 102.597 # 525, 150

    area_grid2[9, 21] = 182.729 # 525, 210

    area_grid2[9, 22] = 182.729 # 525, 220

    area_grid2[10, 10] = 118.303 # 550, 100

    area_grid2[10, 13] = 102.807 # 550, 130

    area_grid2[10, 16] = 96.836 # 550, 160

    area_grid2[10, 17] = 110.92 # 550, 170

    area_grid2[10, 18] = 152.035 # 550, 180

    area_grid2[10, 20] = 203.65 # 550, 200

    area_grid2[10, 21] = 203.65 # 550, 210

    area_grid2[10, 22] = 203.65 # 550, 220

# now ready to interpolate these arrays


#! Creating a function for interpolating the properties at each dt tf case
def interpolate_property(target_dt, target_tf, yielding = False):
    if yielding == False:
        # area_interp = RectBivariateSpline(np.array(unique_dt), np.array(unique_tf), area_grid, kind='linear')
        area_interp = RectBivariateSpline(np.array(unique_dt), np.array(unique_tf), area_grid.transpose(), kx=1, ky=1)  # jwenner workaround
        # damage_interp = interp2d(unique_dt, unique_tf, damage_grid, kind='linear')
        damage_interp = RectBivariateSpline(unique_dt, unique_tf, damage_grid.transpose(), kx=1, ky=1)
    elif yielding == True:
        # area_interp = interp2d(unique_dt, unique_tf, area_grid2, kind='linear')
        area_interp = RectBivariateSpline(unique_dt, unique_tf, area_grid2.transpose(), kx=1, ky=1)
        # damage_interp = interp2d(unique_dt, unique_tf, damage_grid2, kind='linear')
        damage_interp = RectBivariateSpline(unique_dt, unique_tf, damage_grid2.transpose(), kx=1, ky=1)
    else:
        print("Invalid input for yielding case")
        return None, None
    return area_interp(target_dt, target_tf)[0], np.exp(damage_interp(target_dt, target_tf)[0])


#! Scaling damage rate with area under the curve
def find_nearest_case(target_dt, target_tf, stress_data):
    """
    Find the nearest DT-TF case and return its area under curve and damage values.
    
    Parameters:
    -----------
    target_dt : float
        Target temperature difference (DT) value
    target_tf : float
        Target fluid temperature (TF) value
    stress_data : dict
        Dictionary containing the stress data for each case
        
    Returns:
    --------
    area : float
        Area under curve value for the nearest case
    damage : float
        Maximum instantaneous damage value for the nearest case
    """
    # Extract all DT and TF values from file_ids
    dt_values = {}
    tf_values = {}
    
    for file_id in stress_data.keys():
        try:
            dt = int(file_id.split('TD_')[0].split('_')[1])
            tf = int(file_id.split('TF_')[0])
            dt_values[file_id] = dt
            tf_values[file_id] = tf
        except:
            print(f"Could not parse DT-TF values from file_id: {file_id}")
            continue
    
    # Calculate distance to each case
    min_distance = float('inf')
    nearest_file_id = None
    
    for file_id in dt_values.keys():
        dt = dt_values[file_id]
        tf = tf_values[file_id]
        
        # Calculate Euclidean distance
        distance = ((dt - target_dt)**2 + (tf - target_tf)**2)**0.5
        
        # Update if this is the closest case so far
        if distance < min_distance:
            min_distance = distance
            nearest_file_id = file_id
    
    if nearest_file_id is None:
        return 0, 0, float('inf')
    
    # Get the corresponding area and damage values
    area = area_under_curve[nearest_file_id]
    damage = max_instant_dmg_value[nearest_file_id]
    
    return area, damage#, nearest_file_id

####! testing the interpolate_property function
test_interpolate = False
if test_interpolate:
    # First plot all damage interpolations
    for tf in range(300, 551, 25):
        # Create arrays to store results
        dt_values = np.arange(0, 221, 1)  # DT from 0 to 220 in steps of 1
        interpolated_damages = []
        nearest_case_damages = []

        interpolated_damages_yielding = []

        # Collect damages for each DT value
        for dt in dt_values:
            # Get interpolated damage
            try:

                interp_area, interp_damage = interpolate_property(dt, tf, yielding = False)

                interpolated_damages.append(interp_damage)

                interp_area_yielding, interp_damage_yielding = interpolate_property(dt, tf, yielding = True)

                interpolated_damages_yielding.append(interp_damage_yielding)

            except:

                interpolated_damages.append(np.nan)
            
            # Get nearest case damage
            nearest_area, nearest_damage = find_nearest_case(dt, tf, stress_data)
            nearest_case_damages.append(nearest_damage)

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot both curves
        plt.plot(dt_values, interpolated_damages, 'b-', linewidth=2, label = 'Non-Yielding')
        plt.plot(dt_values, nearest_case_damages, 'k--', linewidth=2, label = 'No Interpolation')
        plt.plot(dt_values, interpolated_damages_yielding, 'r-', linewidth=2, label = 'Yielding')

        # Add labels and title
        plt.xlabel('DT (Temperature Difference [°C])')
        plt.ylabel('Damage Rate')
        plt.title(f'Damage Rate vs DT at TF = {tf}°C')
        plt.yscale('log')  # Using log scale since damage values can vary by orders of magnitude
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Show the plot
        plt.show()

    # Then plot all area interpolations
    for tf in range(300, 551, 25):
        # Create arrays to store results
        dt_values = np.arange(0, 221, 1)  # DT from 0 to 220 in steps of 1
        interpolated_areas = []
        nearest_case_areas = []

        # Collect areas for each DT value
        for dt in dt_values:
            # Get interpolated damage
            try:
                interp_area, interp_damage = interpolate_property(dt, tf, yielding = True)
                interpolated_areas.append(interp_area)
            except:
                interpolated_areas.append(np.nan)
            
            # Get nearest case damage
            nearest_area, nearest_damage = find_nearest_case(dt, tf, stress_data)
            nearest_case_areas.append(nearest_area)

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot both curves
        plt.plot(dt_values, interpolated_areas, 'b-', label='Interpolated Area', linewidth=2)
        plt.plot(dt_values, nearest_case_areas, 'r--', label='Nearest Case Area', linewidth=2)

        # Add labels and title
        plt.xlabel('DT (Temperature Difference [°C])')
        plt.ylabel('Area Under Curve')
        plt.title(f'Area Under Curve vs DT at TF = {tf}°C')
        plt.yscale('log')  # Using log scale since damage values can vary by orders of magnitude
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Show the plot
        plt.show()

#! plotting damage and area under the curve on 3d surface plots
# Create the coordinate grids
dt_values = np.arange(0, 221, 5)  # DT from 100 to 220
tf_values = np.arange(300, 551, 5)  # TF from 300 to 550
DT, TF = np.meshgrid(dt_values, tf_values)

# Create arrays to store interpolated values
damage_surface = np.zeros_like(DT)
area_surface = np.zeros_like(DT)

# Fill the surface arrays using the interpolate_property function
for i in range(DT.shape[0]):
    for j in range(DT.shape[1]):
        area_val, dmg_val = interpolate_property(DT[i,j], TF[i,j], yielding = False)
        # Handle potential negative or zero values before taking log
        damage_surface[i,j] = np.log10(dmg_val)
        area_surface[i,j] = area_val

# Create the first figure for damage
fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# Calculate log10 of damage values, handling zeros/negative
# Plot the damage surface with log scale coloring
surf1 = ax1.plot_surface(DT, TF, damage_surface, 
                        cmap='viridis',
                        linewidth=0,
                        antialiased=True)

# Set z-axis limits based on valid log damage values
z_min = np.nanmin(damage_surface)
z_max = np.nanmax(damage_surface)
ax1.set_zlim(z_min, z_max)

# Customize the damage plot
ax1.set_xlabel('DT (Temperature Difference [°C])')
ax1.set_ylabel('TF (Fluid Temperature [°C])')
ax1.set_zlabel('log10(Damage Rate)')
plt.title('Maximum Instantaneous Damage Rate Surface (Log Scale)')
fig1.colorbar(surf1, label='log10(Damage Rate)')

# Add the characteristic line projection
dt_line = np.array([min(dt_values), max(dt_values)])
tf_line = -2.34 * dt_line + 791.5
z_line = z_min * np.ones_like(dt_line)  # Use minimum z value for line projection
ax1.plot(dt_line, tf_line, z_line, 'r--', label='y = -2.34x + 791.5')
ax1.legend()

plt.show()

# Debugging Code:
# Create the second figure for area
# fig2 = plt.figure(figsize=(12, 8))
# ax2 = fig2.add_subplot(111, projection='3d')

# # Plot the area surface
# surf2 = ax2.plot_surface(DT, TF, area_surface,
#                         cmap='viridis',
#                         linewidth=0,
#                         antialiased=True)

# # Customize the area plot
# ax2.set_xlabel('DT (Temperature Difference [°C])')
# ax2.set_ylabel('TF (Fluid Temperature [°C])')
# ax2.set_zlabel('Area Under Curve [MPa·s]')
# plt.title('Area Under Curve Surface')
# fig2.colorbar(surf2, label='Area Under Curve [MPa·s]')

# # Add the characteristic line projection
# z_line = np.min(area_surface) * np.ones_like(dt_line)
# ax2.plot(dt_line, tf_line, z_line, 'r--', label='y = -2.34x + 791.5')
# ax2.legend()

# plt.show()

#! Surface plot of yielding array
# Fill the surface arrays using the interpolate_property function
for i in range(DT.shape[0]):
    for j in range(DT.shape[1]):
        area_val, dmg_val = interpolate_property(DT[i,j], TF[i,j], yielding = True)
        # Handle potential negative or zero values before taking log
        damage_surface[i,j] = np.log10(dmg_val)
        area_surface[i,j] = area_val

# Create the first figure for damage
fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# Calculate log10 of damage values, handling zeros/negative
# Plot the damage surface with log scale coloring
surf1 = ax1.plot_surface(DT, TF, damage_surface, 
                        cmap='viridis',
                        linewidth=0,
                        antialiased=True)

# Set z-axis limits based on valid log damage values
z_min = np.nanmin(damage_surface)
z_max = np.nanmax(damage_surface)
ax1.set_zlim(z_min, z_max)

# Customize the damage plot
ax1.set_xlabel('DT (Temperature Difference [°C])')
ax1.set_ylabel('TF (Fluid Temperature [°C])')
ax1.set_zlabel('log10(Damage Rate)')
plt.title('Maximum Instantaneous Damage Rate Surface (Log Scale) for Yielding Cases')
fig1.colorbar(surf1, label='log10(Damage Rate)')

# Add the characteristic line projection
dt_line = np.array([min(dt_values), max(dt_values)])
tf_line = -2.34 * dt_line + 791.5
z_line = z_min * np.ones_like(dt_line)  # Use minimum z value for line projection
ax1.plot(dt_line, tf_line, z_line, 'r--', label='y = -2.34x + 791.5')
ax1.legend()

plt.show()

#! Surface plot of difference betweenn yielding and non-yielding damage
# Used for debugging
# for i in range(DT.shape[0]):
#     for j in range(DT.shape[1]):
#         area_val, dmg_val = interpolate_property(DT[i,j], TF[i,j], yielding = True)
#         area_val_non_yielding, dmg_val_non_yielding = interpolate_property(DT[i,j], TF[i,j], yielding = False)

#         # Handle potential negative or zero values before taking log
#         damage_val_diff = np.log10(dmg_val_non_yielding) - np.log10(dmg_val)
#         damage_surface[i,j] = damage_val_diff

# # Create the first figure for damage
# fig1 = plt.figure(figsize=(12, 8))
# ax1 = fig1.add_subplot(111, projection='3d')

# # Calculate log10 of damage values, handling zeros/negative
# # Plot the damage surface with log scale coloring
# surf1 = ax1.plot_surface(DT, TF, damage_surface, 
#                         cmap='viridis',
#                         linewidth=0,
#                         antialiased=True)

# # Set z-axis limits based on valid log damage values
# z_min = np.nanmin(damage_surface)
# z_max = np.nanmax(damage_surface)
# ax1.set_zlim(z_min, z_max)

# # Customize the damage plot
# ax1.set_xlabel('DT (Temperature Difference [°C])')
# ax1.set_ylabel('TF (Fluid Temperature [°C])')
# ax1.set_zlabel('log10(Damage Rate)')
# plt.title('Difference in Damage Rates between Yielding and Non-Yielding Cases (Log Scale)')
# fig1.colorbar(surf1, label='log10(Damage Rate)')

# # Add the characteristic line projection
# dt_line = np.array([min(dt_values), max(dt_values)])
# tf_line = -2.34 * dt_line + 791.5
# z_line = z_min * np.ones_like(dt_line)  # Use minimum z value for line projection
# ax1.plot(dt_line, tf_line, z_line, 'r--', label='y = -2.34x + 791.5')
# ax1.legend()

# plt.show()


####! calculating the accrued damage for each file_id
accrued_damage = {file_id: [] for file_id in stress_data.keys()}
accrued_damage_non_scaled = {file_id: [] for file_id in stress_data.keys()}
percent_error_scaling_damage = {file_id: [] for file_id in stress_data.keys()}
percent_error_nonscaling_damage = {file_id: [] for file_id in stress_data.keys()}
nearest_case_history = {file_id: [] for file_id in stress_data.keys()}
damage_history = {file_id: [] for file_id in stress_data.keys()}
non_scaled_damage_history = {file_id: [] for file_id in stress_data.keys()}
interpolated_accrued_damage = {file_id: [] for file_id in stress_data.keys()}
percent_error_interpolated_damage = {file_id: [] for file_id in stress_data.keys()}
interpolated_damage_history = {file_id: [] for file_id in stress_data.keys()}
interpolated_dmg_scaled_history = {file_id: [] for file_id in stress_data.keys()}
interpolated_dmg_scaled_accrued = {file_id: [] for file_id in stress_data.keys()}
percent_error_interpolated_dmg_scaled = {file_id: [] for file_id in stress_data.keys()}
for file_id in stress_data.keys():
    #finding the last day temperature profile
    sub = substep_num[file_id]
    per = period_num[file_id]
    temperature_profile = temps[file_id][-per*sub:]
    
    #calculating properties
    area_under_curve_file = area_under_curve[file_id]
    tf = int(file_id.split('TF_')[0])
    dt = int(file_id.split('TD_')[0].split('_')[1])
    
    #initializing accrued damage
    accrued_damage[file_id] = 0
    accrued_damage_non_scaled[file_id] = 0
    interpolated_accrued_damage[file_id] = 0
    interpolated_dmg_scaled_accrued[file_id] = 0
    # Initialize the list for this file_id
    nearest_case_history[file_id] = []
    
    #calculating accrued damage
    if (dt >=140) or (dt==130 and tf>400) or (dt==120 and tf>475) or (dt==110 and tf>500):
        yielding = True
    else:
        yielding = False
    for temp in temperature_profile:
        nearest_area, nearest_dmg = find_nearest_case(temp-tf, tf, stress_data)
        interpolated_area, interpolated_dmg = interpolate_property(temp-tf, tf, yielding)
        # retrieving properties of the nearest case
        damage_rate = nearest_dmg*3600/sub
        interpolated_damage_rate = interpolated_dmg*3600/sub

        #scaling the damage rate
        scaled_damage_rate = damage_rate * (area_under_curve_file/ nearest_area)
        #print("damage rate scaling factor: ", area_under_curve_file/area_under_curve_nearest)

        # accruing damage
        if temp < tf:
            scaled_damage_rate = 0
            damage_rate = 0
        elif temp >= tf:
            accrued_damage[file_id] = scaled_damage_rate + accrued_damage[file_id]
            accrued_damage_non_scaled[file_id] = damage_rate + accrued_damage_non_scaled[file_id]
            interpolated_accrued_damage[file_id] = interpolated_damage_rate + interpolated_accrued_damage[file_id]
            interpolated_dmg_scaled_accrued[file_id] = interpolated_damage_rate*(area_under_curve_file/interpolated_area) + interpolated_dmg_scaled_accrued[file_id]
        damage_history[file_id].append(scaled_damage_rate)
        non_scaled_damage_history[file_id].append(damage_rate)
        interpolated_damage_history[file_id].append(interpolated_damage_rate)
        #print("interpolated dmg: ", interpolated_accrued_damage[file_id])
        interpolated_dmg_scaled_history[file_id].append(interpolated_damage_rate*(area_under_curve_file/interpolated_area))
        #print("interpolated scaled dmg : ", interpolated_dmg_scaled_accrued[file_id])

    # assessing accuracy
    fea_damage =sum(time_damages[file_id][-14*sub:])
    percent_error_scaling_damage[file_id] = (accrued_damage[file_id]-fea_damage)/fea_damage
    percent_error_nonscaling_damage[file_id] = (accrued_damage_non_scaled[file_id]-fea_damage)/fea_damage
    percent_error_interpolated_damage[file_id] = (interpolated_accrued_damage[file_id]-fea_damage)/fea_damage
    percent_error_interpolated_dmg_scaled[file_id] = (interpolated_dmg_scaled_accrued[file_id]-fea_damage)/fea_damage
#------------ Create scatter plot to display percent error scaling damage ------------
plt.figure(figsize=(12, 10))

# Extract values for plotting
tf_values = []
dt_values = []
percent_errors = []

for file_id in stress_data.keys():
    if file_id in percent_error_scaling_damage:
        # Extract TF and DT from file_id
        tf = int(file_id.split('TF_')[0])
        dt = int(file_id.split('TD_')[0].split('_')[1])
        
        tf_values.append(tf)
        dt_values.append(dt)
        
        # Get the percent error value
        error_value = percent_error_interpolated_damage[file_id]
        #print(error_value)
        percent_errors.append(error_value)

# Create empty scatter plot to set the axes
plt.scatter(dt_values, tf_values, c='none', alpha=0)

# Define text colors based on error magnitude
for dt, tf, error in zip(dt_values, tf_values, percent_errors):
    error_percent = float(error * 100)  # Convert to float and percentage
    
    # Choose color based on error value
    if abs(error_percent) < 10:
        text_color = 'green'
    elif abs(error_percent) < 25:
        text_color = 'orange'
    else:
        text_color = 'red'
    
    # Add text label with the percent error
    plt.text(dt, tf, f'{round(error_percent)}', 
            color=text_color,
            ha='center', va='center',
            fontweight='bold',
            fontsize=10)

# Add the characteristic line
dt_range = np.array([min(dt_values), max(dt_values)])
tf_line = -2.34 * dt_range + 791.5
plt.plot(dt_range, tf_line, 'k--', label='y = -2.34x + 791.5')

# Add labels and title
plt.xlabel('DT (Temperature Difference [°C])')
plt.ylabel('TF (Fluid Temperature [°C])')
plt.title('Percent Error Using Instantaneous Damage Rate Interpolation Method')
plt.grid(True, alpha=0.3)
plt.legend()

# Create a legend for the color coding
legend_elements = [
    plt.Line2D([0], [0], marker='$-$', color='green', linestyle='None', 
               markersize=15, label='Error < 10%'),
    plt.Line2D([0], [0], marker='$-$', color='orange', linestyle='None', 
               markersize=15, label='Error 10-25%'),
    plt.Line2D([0], [0], marker='$-$', color='red', linestyle='None', 
               markersize=15, label='Error > 25%'),
    plt.Line2D([0], [0], color='k', linestyle='--', label='y = -2.34x + 791.5')
]
plt.legend(handles=legend_elements, loc='best')

plt.show()


#! Debugging: showing stress profile, temp, and damage for each file id
view_each_case = False
if view_each_case:
    for file_id in stress_data.keys():
        dt = int(file_id.split('TD_')[0].split('_')[1])
        if dt < 140:
            continue
        
        sub = substep_num[file_id]
        file_stresses = stresses[file_id][-14*sub:]
        temp_prof = temps[file_id][-14*sub:]
        proper_damage_rate = time_damages[file_id][-14*sub:]

        # retrieving damage arrays
        non_scaled_damage_rate = non_scaled_damage_history[file_id]
        #damage_hist = damage_history[file_id]
        interpolated_dmg = interpolated_damage_history[file_id]
        #interpolated_dmg_scaled = interpolated_dmg_scaled_history[file_id]

        # Create figure and primary axis
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Create two more axes sharing the same x-axis
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # Offset the right spine of ax3 by 60 points
        ax3.spines['right'].set_position(('outward', 60))

        # Plot temperature on first axis (left)
        line1 = ax1.plot(temp_prof, label="Temperature Profile", color="black", linestyle="dotted")
        ax1.set_ylabel('Temperature [°C]', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Plot stresses on second axis (first right)
        line2 = ax2.plot(file_stresses, label="Stress Profile", color="red", linestyle="dashed")
        ax2.set_ylabel('Stress [MPa]', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Plot damage on third axis (second right)
        line3 = ax3.plot(non_scaled_damage_rate, label="Non-Interpolated Damage", color="orange", linestyle="dotted")
        line4 = ax3.plot(proper_damage_rate, label="FEA Damage", color="blue", linestyle="dotted")
        #line5 = ax3.plot(damage_hist, label="Scaled, Non-Interpolated Damage", color="red", linestyle="solid")
        line6 = ax3.plot(interpolated_dmg, label="Interpolated Damage", color="green", linestyle="solid")
        #line7 = ax3.plot(interpolated_dmg_scaled, label="Scaled, Interpolated Damage ", color="purple", linestyle="solid")
        ax3.set_ylabel('Damage Rate', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')

        # Combine all lines for the legend
        lines = line1 + line2 + line4 + line3 +line6
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

        plt.title(f"Temperature, Stress, and Damage Profiles for {file_id}")
        plt.tight_layout()
        plt.show()

    
#! insert plot of fea damage, interpolated damage

# Create main scatter plot
fig, ax = plt.subplots(figsize=(12, 10))

# Lists to store values
tf_values = []
dt_values = []

# Extract TF and DT values from file_ids
for file_id in stress_data.keys():
    tf_num = int(file_id.split('TF_')[0])
    dt_num = int(file_id.split('TD_')[0].split('_')[1])
    
    tf_values.append(tf_num)
    dt_values.append(dt_num)

# Create main scatter plot with black dots
scatter = ax.scatter(dt_values, tf_values, c='black', s=100)

# Now add inset plots for each point
for dt, tf, file_id in zip(dt_values, tf_values, stress_data.keys()):
    # Create small inset axes for this point
    inset_ax = inset_axes(ax,
                         width=0.7,  # width of inset
                         height=0.5,  # height of inset
                         loc='center',
                         bbox_to_anchor=(dt, tf, 0.0, 0.0),
                         bbox_transform=ax.transData)
    
    # Plot the three damage histories
    sub = substep_num[file_id]
    inset_ax.plot(time_damages[file_id][-14*sub:], color='blue', linewidth=0.5)
    inset_ax.plot(interpolated_damage_history[file_id], color='red', linewidth=0.5)
    
    # Remove ticks and labels
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])

# Main plot formatting
ax.set_xlabel('DT (Temperature Difference [°C])')
ax.set_ylabel('TF (Fluid Temperature [°C])')
ax.set_title('Damage History Comparison')

# Create custom legend handles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', label='FEA'),
    Line2D([0], [0], color='red', label='Predicted')
]
ax.legend(handles=legend_elements, loc='upper right')

# Add the characteristic line y = -2.34x + 791.5
dt_range = np.array([min(dt_values), max(dt_values)])
tf_line = -2.34 * dt_range + 791.5
ax.plot(dt_range, tf_line, 'r--', label='y = -2.34x + 791.5')

plt.tight_layout()
plt.show()

#! visualizing damage for each iso-tf
# Primarily used for debugging

# file_damage_data = []
# for file_id in stress_data.keys():
#     dt = int(file_id.split('TD_')[0].split('_')[1])
#     tf = int(file_id.split('TF_')[0])
#     sub = substep_num[file_id]
#     dmgs_per_substep = time_damages[file_id][-14*sub:]
#     dmgs_per_substep = dmgs_per_substep[0:len(dmgs_per_substep)//2]
#     file_damage_data.append((dt,tf,dmgs_per_substep))

# # Group data by TF values
# tf_groups = {}
# for dt, tf, dmgs in file_damage_data:
#     if tf not in tf_groups:
#         tf_groups[tf] = []
#     tf_groups[tf].append((dt, dmgs))

# # Create a plot for each TF value
# for tf in sorted(tf_groups.keys()):
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Plot damage rates for each DT in this TF group
#     for dt, dmgs in sorted(tf_groups[tf]):
#         # Calculate damage rate (difference between consecutive points)
#         damage_rate = np.diff(dmgs)
#         time_points = np.arange(len(damage_rate))
        
#         ax.plot(time_points, damage_rate, label=f'DT = {dt}°C')
    
#     ax.set_xlabel('Time Step')
#     ax.set_ylabel('Damage Rate')
#     ax.set_title(f'Damage Rates for TF = {tf}°C')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax.grid(True)
    
#     plt.tight_layout()
#     plt.show()

#! Comparing the difference between the 20th day FEA stress minus the 20th day NB prediction
# Used for Jacobs Research
# --- Plot: FEA 20th day max stress minus NB 30-day prediction (color-coded) ---

# fig, ax = plt.subplots(figsize=(12, 10))
fig, ax = plt.subplots(tight_layout=True)

tf_values = []
dt_values = []
diff_values = []
fea_firsts  =[]
fea_lasts   =[]
nb_lasts    =[]

for file_id in stress_data.keys():
    # Extract TF and DT from file_id
    tf = int(file_id.split('TF_')[0])
    dt = int(file_id.split('TD_')[0].split('_')[1])
    t_total = tf + dt + 273.15  # Kelvin

    # Get FEA last day max stress (MPa)
    fea_last = max_stress_FEA[file_id][-1]

    # Get FEA first day max stress (MPa)
    fea_first = max_stress_FEA[file_id][0]

    # Calculate Young's modulus at t_total (K)
    E = get_Youngs_Modulus(t_total, material) * 1e6  # [Pa]

    # NB 30-day prediction (Pa)
    nb_20_pred_pa = nb.norton_bailey_gonzalez(
        fea_first * 1e6,   # convert MPa to Pa
        20 * 60 * 60,      # 20 days in seconds
        E,
        t_total            # T in K
    )
    # Convert NB prediction to MPa for comparison
    nb_20_pred_mpa = -nb_20_pred_pa / 1e6 + fea_first  # as in your code

    # Calculate difference (MPa)
    diff = fea_last - nb_20_pred_mpa

    tf_values.append(tf)
    dt_values.append(dt)
    diff_values.append(diff)
    fea_firsts.append(fea_first)
    fea_lasts.append(fea_last)
    nb_lasts.append(nb_20_pred_mpa)

# Create scatter plot with color scale
fontsize=16
sc = ax.scatter(dt_values, tf_values, c=diff_values, cmap='viridis', s=100)
ax.set_xlabel('Temperature Difference (°C)', fontsize=fontsize)
ax.set_ylabel('Fluid Temperature (°C)', fontsize=fontsize)
# plt.title('FEA 20th Day Max Stress - NB 20-Day Prediction (MPa)')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=fontsize)
# Add colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(label='Difference (MPa)', fontsize=fontsize)

ax.legend()
plt.savefig('imgs/FEA_versus_norton_bailey.png', dpi=300)
# plt.tight_layout()
plt.show()

print(f'discrepancy (FEA - NB) is {np.array(diff_values).min()}--{np.array(diff_values).max()}')

print('pause')