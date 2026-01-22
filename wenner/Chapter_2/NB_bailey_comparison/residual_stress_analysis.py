"""
This file was used by Frankie Iovinelli to analyze the residual stress vs elastic minus plastic/actual stress relationship.
This was a key file in developing the final version of the instantaneous damage model.
1. The bestfit line was found for the residual stress vs elastic minus plastic/actual stress relationship.
2. The elastic stress and residual stress lookup tables were created here.
3. The residual stress prediction model was tested against the FEA data.

last modified: 6/2/2025
Built by: Frankie Iovinelli
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import cm
from sklearn.metrics import r2_score
import BoundaryConditions as bc
from scipy.interpolate import interp2d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import LifetimeCalculations as lc



#! Define functions
def get_tf_td(id_str):
    # Example id_str: "ELASTIC_650TF_150TD_10SUB_20PER" or "650TF_150TD_10SUB_20PER"
    tf = id_str.split('_')[0]  # e.g., "650TF"
    td = id_str.split('_')[1]  # e.g., "150TD"
    return tf, td

def find_matching_elastic_id(plastic_id, elastic_max_crown_stress):
    #print("Plastic ID: ", plastic_id)
    tf, td = get_tf_td(plastic_id)
    for elastic_id in elastic_max_crown_stress.keys():
        if elastic_id.startswith(f"ELASTIC_{tf}_{td}"):
            #print("Elastic ID: ", elastic_id,"\n")
            return elastic_id
    return None  # Not found

def plot_residual_vs_elastic_yield(x_all, y_all, dt_all, tf_all, day_all, total_temp_all, color_by="tf"):
    """
    Scatterplot of residual stress vs elastic - yield, colored by dt, tf, day, or total temperature.
    Uses Reds for dt/tf, Purples for day.
    Includes a dashed black y=x line.
    """
    y_pred = x_all
    r2 = r2_score(y_all, y_pred)

    if color_by == "dt":
        c = dt_all
        cmap = plt.cm.Reds
        norm = mpl.colors.Normalize(vmin=min(dt_all), vmax=max(dt_all))
        label = "dT (°C)"
    elif color_by == "tf":
        c = tf_all
        cmap = plt.cm.Reds
        norm = mpl.colors.Normalize(vmin=min(tf_all), vmax=max(tf_all))
        label = "TF (°C)"
    elif color_by == "day":
        c = day_all
        cmap = plt.cm.Reds
        norm = mpl.colors.Normalize(vmin=min(day_all), vmax=max(day_all))
        label = "Day"
    elif color_by == "total":
        c = total_temp_all
        cmap = plt.cm.Reds
        norm = mpl.colors.Normalize(vmin=min(total_temp_all), vmax=max(total_temp_all))
        label = "Total Temperature (°C)"
    else:
        raise ValueError("color_by must be 'dt', 'tf', 'day', or 'total'")

    fig, ax = plt.subplots()
    sc = ax.scatter(x_all, y_all, c=c, cmap=cmap, norm=norm, s=40, edgecolor='none', alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(axis='both', labelsize=12)
    cbar.set_label(label=label, fontsize=14)
    ax.set_xlabel("Residual Stress (MPa)", fontsize=14)
    ax.set_ylabel("Elastic Stress - Peak Actual Stress (MPa)", fontsize=14)
    # ax.set_title(f"Elastic - Peak vs Residual Stress [Colored by {label}]")

    # Plot y = x line (slope = 1, dashed black)
    min_val = min(min(x_all), min(y_all))
    max_val = max(max(x_all), max(y_all))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x (slope=1)')

    # Plot the now bestfit line
    x_data = [0,225.6]
    y_data = [-3.6,207.15]
    plt.plot(x_data, y_data, 'k-', label='Bestfit Line from Data')

    # Assessing R^2 from the bestfit line
    y_pred = [x*(0.9337) -3.6 for x in x_all]
    r2_bestfit = r2_score(y_all, y_pred)

    # Add a statistics box in the bottom right
    stats_text = f'$R^2 (m=1) = {r2:.4f}$\n $R^2 (bestfit) = {r2_bestfit:.4f}$'
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=14,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )

    ax.legend(fontsize=14)
    plt.savefig('imgs/elastic_vs_residual_default', dpi=300)
    plt.show()

def calculate_actual_stress(sigma_elastic, sigma_residual, m, b):
    sigma_actual = sigma_elastic - m*sigma_residual-b
    return sigma_actual

material = "A230" # or 617
if(material == "A230"):
    base_path = "20day_stresses_A230"
else:
    print("No material selected")
    exit()
# note: do not have elastic data for A617, so this code is only for A230

def interpolate_elastic_stress(tf, dt, elastic_max_crown_stress):
    elastic_interp = interp2d(unique_dt, unique_tf, elastic_max_crown_stress, kind='linear')
    return elastic_interp(dt, tf)[0]


# Find CSV files
path = os.path.join(base_path, "*.csv")
csv_files = glob.glob(path)
# Filter CSV files to only include those with "stresses" in the filename
csv_files = [f for f in csv_files if "stresses" in f.lower()]

# Create dictionary to store the data
stress_data = {}

# Create sets to track files
processed_files = set()
failed_files = set()
duplicate_ids = set()

#! Processing plastic cases
#print("\nProcessing files:")
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
        
        #print(f"Processed: {filename} → {file_id}")
        
    except Exception as e:
        failed_files.add(filename)
        print(f"Error processing {filename}: {e}")

# Debugging Code:
# print(f"\nSummary:")
# print(f"Total CSV files found: {len(csv_files)}")
# print(f"Successfully processed: {len(processed_files)}")
# print(f"Failed to process: {len(failed_files)}")
# print(f"Duplicate IDs found: {len(duplicate_ids)}")

if failed_files:
    print("\nFailed files:")
    for f in failed_files:
        print(f"  {f}")

if duplicate_ids:
    print("\nDuplicate IDs:")
    for id in duplicate_ids:
        print(f"  {id}")

#! Processing elastic cases
# Find CSV files
elastic_path = 'elastic_stresses_A230'
elastic_path = os.path.join(elastic_path, "*.csv")
elastic_csv_files = glob.glob(elastic_path)
# Filter CSV files to only include those with "stresses" in the filename
elastic_csv_files = [f for f in elastic_csv_files if "stresses" in f.lower()]

# Create dictionary to store the data
elastic_stress_data = {}

# Create sets to track files
processed_files = set()
failed_files = set()
duplicate_ids = set()

#print("\nProcessing files:")
for file in elastic_csv_files:
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
        file_id = f"ELASTIC_{tf_num}TF_{td_num}TD_{substep_num}SUB_{period_num}PER"
        
        # Check if this ID already exists
        if file_id in elastic_stress_data:
            duplicate_ids.add(file_id)
            print(f"WARNING: Duplicate ID found: {file_id}")
            print(f"  Original file: {processed_files[file_id]}")
            print(f"  New file: {filename}")
    
        # Read the CSV and store data
        df_elastic = pd.read_csv(file)
        elastic_stress_data[file_id] = df_elastic
        processed_files.add(filename)
        
        #print(f"Processed: {filename} → {file_id}")
        
    except Exception as e:
        failed_files.add(filename)
        print(f"Error processing {filename}: {e}")

# Debugging Code:
# print(f"\nSummary:")
# print(f"Total Elastic CSV files found: {len(elastic_csv_files)}")
# print(f"Successfully processed: {len(processed_files)}")
# print(f"Failed to process: {len(failed_files)}")
# print(f"Duplicate IDs found: {len(duplicate_ids)}")

if failed_files:
    print("\nFailed files:")
    for f in failed_files:
        print(f"  {f}")

if duplicate_ids:
    print("\nDuplicate IDs:")
    for id in duplicate_ids:
        print(f"  {id}")

#Now we have all the elastic and plastic data now
#! Comparing the residual stress to the elastic - plastic data
# First lets get a list of the residual stress data
elastic_max_crown_stress = {}
for elastic_id in elastic_stress_data.keys():
    elastic_crown_stress = elastic_stress_data[elastic_id]["vmRoutsAll"]
    max_elastic_crown_stress = elastic_crown_stress.max()
    elastic_max_crown_stress[elastic_id] = max_elastic_crown_stress

# Now lets get a list of the residual stress data
residual_stress = {plastic_id:[] for plastic_id in stress_data.keys()}
yield_stress = {}
elastic_minus_yield = {plastic_id:[] for plastic_id in stress_data.keys()}
elastic_minus_peak = {plastic_id:[] for plastic_id in stress_data.keys()}

for plastic_id in stress_data.keys():
    plastic_crown_stress = stress_data[plastic_id]["vmRoutsAll"]
    sub= int(plastic_id.split('_')[2].replace('SUB', ''))
    per = int(plastic_id.split('_')[3].replace('PER', ''))
    steps_per_day = sub*per
    days = round(len(plastic_crown_stress)/steps_per_day)

    matching_elastic_id = find_matching_elastic_id(plastic_id, elastic_max_crown_stress)

    for day in range(days): # for day in all_stress_data/day_length
        residual_stress[plastic_id].append(plastic_crown_stress[(day+1)*steps_per_day])

        # finding peak stress for the day
        peak_stress = plastic_crown_stress[(day)*steps_per_day:((day+1)*steps_per_day)].max()
        elastic_minus_peak[plastic_id].append(elastic_max_crown_stress[matching_elastic_id] - peak_stress)
        if day == 0:
            first_day_stresses = plastic_crown_stress[:steps_per_day]
            yield_stress[plastic_id] = max(first_day_stresses)

    if matching_elastic_id is not None:
        elastic_minus_yield[plastic_id].append(elastic_max_crown_stress[matching_elastic_id] - 250)#yield_stress[plastic_id])
    else:
        print(f"No matching elastic case found for {plastic_id}")
    

# statistical analysis of all yield stresses: is each case the same? What is the variance and the mean? This is to check if the yield stress is the same for all cases
yield_stress_list = list(yield_stress.values())
len(yield_stress_list)
mean_yield_stress = sum(yield_stress_list) / len(yield_stress_list)
total_residual_stress_data_points = sum(len(residual_stress[plastic_id]) for plastic_id in stress_data.keys())
print("total number of data points: ", total_residual_stress_data_points, " should be 124*20 = 2480")

#! PLOTTING THE ELASTIC - YIELD VS RESIDUAL STRESS

# First identify deviant cases based on the residual stress vs elastic-peak relationship
deviant_file_ids = []

# Lists to store values for plotting
x_all = []
y_all = []
dt_all = []
tf_all = []
day_all = []
total_temp_all = []
deviant_cases = []
deviant_threshold = 50  # x MPa threshold for deviation
for plastic_id in stress_data.keys():
    # Extract dT, TF, and number of days
    dt = int(plastic_id.split('_')[1].replace('TD', ''))
    tf = int(plastic_id.split('_')[0].replace('TF', ''))

    

    # Get the list of elastic_minus_yield and residual_stress for this case
    emp_list = elastic_minus_peak[plastic_id]
    res_list = residual_stress[plastic_id]
    
    # Check if this is a deviant case
    is_deviant = False
    for r, emp in zip(res_list, emp_list):
        predicted_emp = r * 0.9337 - 3.5
        if abs(predicted_emp - emp) > deviant_threshold:
            print(f"Deviant case found: {plastic_id}")
            deviant_file_ids.append(plastic_id)
            is_deviant = True
            break
    
    # Skip this plastic_id if it's a deviant case
    if is_deviant:
        continue

    # If not deviant, add the data points
    n_days = 20
    for day in range(n_days):
        x_all.append(res_list[day])
        y_all.append(emp_list[day])
        dt_all.append(dt)
        tf_all.append(tf)
        total_temp_all.append(dt + tf)
        day_all.append(day)


# Now plot using your function
plot_residual_vs_elastic_yield(x_all, y_all, dt_all, tf_all, day_all, total_temp_all, color_by="total")  # or "dt" or "day"
#! set m and b (THESE ARE FOUND FROM THE FIRST PLOT)
m = 0.9337
b = -3.5

#! Testing this damage model on the nonconstant thermal loading case
non_constant_case_dir = "stresses&strains_dframe_A230_300TF_100DT_20DAYS_450TF_160DT_5DAYS_300TF_100DT_5DAYS_0.4289R_30substeps_30days_14period_p11.csv"
non_constant_case_df = pd.read_csv(non_constant_case_dir)
fontsize=14
plt.figure()
times   =np.linspace(0,30,non_constant_case_df["vmRoutsAll"].size)
plt.plot(times,non_constant_case_df["vmRoutsAll"])
plt.xlabel('days', fontsize=fontsize)
plt.ylabel('crown stress (MPa)', fontsize=fontsize)
plt.savefig('imgs/residual_vs_fea_test_stress_trace', dpi=300)
plt.show()

ncloading_stress = non_constant_case_df["vmRoutsAll"].to_numpy()
day_steps = 14*30 #! hardcoded
max_daily_stresses = []
residual_stresses = []
initial_loading_case = find_matching_elastic_id("300TF_100TD_20SUB_14PER", elastic_max_crown_stress)
higher_loading_case = find_matching_elastic_id("450TF_160TD_20SUB_14PER", elastic_max_crown_stress)
print(initial_loading_case, higher_loading_case)
for day in range(30):
    day_stress = ncloading_stress[day*day_steps:(day+1)*day_steps]
    max_stress_day = day_stress.max()
    max_daily_stresses.append(max_stress_day)
    residual_stress_day = day_stress[-1]
    residual_stresses.append(residual_stress_day)

# predicting the change in the case from higher to lower
residual_end_of_20 = residual_stresses[19]
max_stress_end_of_20 = max_daily_stresses[19]

sigma_actual_21 = calculate_actual_stress(elastic_max_crown_stress[initial_loading_case], residual_end_of_20, m, b)
print(sigma_actual_21-max_daily_stresses[19])

# predicting the max stress each day
sigma_actual = []
for day in range(30):
    if day <20:
        sigma_actual_day = calculate_actual_stress(elastic_max_crown_stress[initial_loading_case], residual_stresses[day], m, b)
        sigma_actual.append(sigma_actual_day)
    if day>=20 and day<25:
        sigma_actual_day = calculate_actual_stress(elastic_max_crown_stress[higher_loading_case], residual_stresses[day], m, b)
        sigma_actual.append(sigma_actual_day)
    if day>=25:
        sigma_actual_day = calculate_actual_stress(elastic_max_crown_stress[initial_loading_case], residual_stresses[day], m, b)
        sigma_actual.append(sigma_actual_day)

plt.figure()
plt.plot(sigma_actual, label="Residual Model-Predicted", color="red", marker='.')
plt.plot(max_daily_stresses, label="FEA", color="blue", marker='.')
plt.xlabel("Day", fontsize=fontsize)
plt.ylabel("Max Crown Stress (MPa)", fontsize=fontsize)
plt.title("Max Stress Comparison", fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.savefig('imgs/residual_vs_fea_300-100_450-160', dpi=300)
plt.show()

# plotting the residual stresses and the actual stresses
plt.figure()
plt.plot(residual_stresses, label="Residual Stress", color="black")
plt.plot(sigma_actual, label="Actual Stress", color="red")
plt.axhline(y=elastic_max_crown_stress[initial_loading_case], label="Elastic Stress", color="purple", linestyle="--")
plt.axhline(y=elastic_max_crown_stress[higher_loading_case], label="Elastic Stress", color="green", linestyle="--")
plt.xlabel("Day")
plt.ylabel("Stress (MPa)")
plt.title("Residual Stress and Actual Stress Comparison")
plt.legend()
plt.show()

#! calculating the residual stress for each tf dt combination

# Plotting max residual stress for each tf/dt combination
dt_vals = []
tf_vals = []
max_residuals = []

for plastic_id in stress_data.keys():
    # Extract dT and TF from the file ID
    dt = int(plastic_id.split('_')[1].replace('TD', ''))
    tf = int(plastic_id.split('_')[0].replace('TF', ''))
    # Get the max residual stress for this file
    if plastic_id in residual_stress and len(residual_stress[plastic_id]) > 0:
        max_res = max(residual_stress[plastic_id])
        dt_vals.append(dt)
        tf_vals.append(tf)
        max_residuals.append(max_res)

plt.figure()
sc = plt.scatter(dt_vals, tf_vals, c=max_residuals, cmap='Reds', s=60)
plt.colorbar(sc, label='Max Residual Stress (MPa)')
plt.xlabel('dT (°C)')
plt.ylabel('TF (°C)')
plt.title('Max Residual Stress for Each TF/dT Combination')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Create a DataFrame with TF, DT, and max elastic stress
elastic_stress_summary = []
for elastic_id in elastic_max_crown_stress.keys():    
    # Split and filter out any empty strings
    components = [x for x in elastic_id.split('_') if x]
    
    # Extract TF and DT from the elastic_id
    tf = int(components[1].replace('TF', ''))  # Gets XXX from XXXTF
    
    # Find the component that contains 'TD'
    td_component = next(x for x in components if 'TD' in x)
    dt = int(td_component.replace('TD', ''))  # Gets YYY from YYYTD
    
    max_stress = elastic_max_crown_stress[elastic_id]
    
    elastic_stress_summary.append({
        'TF': tf,
        'DT': dt,
        'Max_Elastic_Stress': max_stress
    })

# Convert to DataFrame and save
elastic_stress_df = pd.DataFrame(elastic_stress_summary)
elastic_stress_df.to_csv("elastic_stress_lookup_table.csv", index=False)

#! Making a csv for all residual stresses 
#! First fill unknown values. do this plotting residual stress vs dt for each tf and determining the best residual stress to fill the missing values
unique_tf_values = sorted(set(int(plastic_id.split('_')[0].replace('TF', '')) for plastic_id in stress_data.keys()))

# For each TF value, create and show a plot
for tf in unique_tf_values:
    # Create a new figure for each TF
    #plt.figure(figsize=(10, 5))
    
    # Get all DT values and corresponding residual stresses for this TF
    dt_values = []
    residual_stresses = []
    
    for plastic_id in stress_data.keys():
        current_tf = int(plastic_id.split('_')[0].replace('TF', ''))
        if current_tf == tf:
            dt = int(plastic_id.split('_')[1].replace('TD', ''))
            if plastic_id in residual_stress and len(residual_stress[plastic_id]) > 0:
                max_res = max(residual_stress[plastic_id])
                dt_values.append(dt)
                residual_stresses.append(max_res)
    
    # Sort the data by DT for better visualization
    sorted_indices = np.argsort(dt_values)
    dt_values = np.array(dt_values)[sorted_indices]
    residual_stresses = np.array(residual_stresses)[sorted_indices]
    
    # # Used to find gaps in data: Plot the data
    # plt.plot(dt_values, residual_stresses, 'o-', label=f'TF = {tf}°C')
    # plt.xlabel('DT (°C)')
    # plt.ylabel('Max Residual Stress (MPa)')
    # plt.title(f'Residual Stress vs DT for TF = {tf}°C')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    
    # # Show the plot
    # plt.show()

# Create a DataFrame with TF, DT, and max residual stress
residual_stress_summary = []
for plastic_id in residual_stress.keys():
    # Extract TF and DT from the plastic_id
    # Format is "XXXTF_YYYTD_ZZSUB_WWPER"
    tf = int(plastic_id.split('_')[0].replace('TF', ''))  # Gets XXX from XXXTF
    dt = int(plastic_id.split('_')[1].replace('TD', ''))  # Gets YYY from YYYTD
    
    # Get the max residual stress for this case
    if plastic_id in residual_stress and len(residual_stress[plastic_id]) > 0:
        max_res = max(residual_stress[plastic_id])
        
        residual_stress_summary.append({
            'TF': tf,
            'DT': dt,
            'Max_Residual_Stress': max_res
        })
manually_interpolated_data = [
(210,325,167.20),
(200,400,157.57),
(150, 450, 58.92),
(170,475,134.26),
(200,475, 236.88),
(210,500,279.98),
(120,525,46.27),
(130,525,75.86),
(150,525,138.65),
(210,525,283.80),
(220,525,283.80),
(130,550,108.17),
(160,550,200.00),
(170,550,227.27),
(180,550, 255.00),
(200,550,283.80),
(210,550,283.80),
(220,550,283.80),
(100,550,22.30)
]

# Add the additional data to residual_stress_summary
for dt, tf, stress in manually_interpolated_data:
    residual_stress_summary.append({
        'TF': tf,
        'DT': dt,
        'Max_Residual_Stress': stress
    })

# Convert to DataFrame and save
residual_stress_df = pd.DataFrame(residual_stress_summary)
residual_stress_df.to_csv("residual_stress_lookup_table.csv", index=False)

#! plotting the plastic (fea) stresses vs the predicted stresses using my model
# must first fill the elastic stress grid by extrapolating the data from an arbitrary, low dt case , say 100c
temps = {file_id:[] for file_id in stress_data.keys()}
file_tuples = []
substep_num = {}
period_num = {}
for file_id in elastic_stress_data.keys():
    #print("File ID: ", file_id)
    # Split and filter out any empty strings
    components = [x for x in file_id.split('_') if x]
    tf_num = int(components[1].replace('TF', ''))  # Gets XXX from XXXTF
    #print("TF: ", tf_num)
    # Find the component that contains 'TD' instead of using components[1]
    td_component = next(x for x in components if 'TD' in x)
    td_num = int(td_component.replace('TD', ''))  # Gets YYY from YYYTD
    #print("TD: ", td_num)
    file_tuples.append((td_num, tf_num, elastic_max_crown_stress[file_id]))
    
    # Get both SUB and PER components
    sub_component = next(x for x in components if 'SUB' in x)
    per_component = next(x for x in components if 'PER' in x)
    
    # Store both values in their respective dictionaries
    substep_num[file_id] = int(sub_component.replace('SUB', ''))
    period_num[file_id] = int(per_component.replace('PER', ''))
    
    # getting a temperature profile to use later
    days = 20
    times = np.linspace(0,days*period_num[file_id],days*period_num[file_id]*substep_num[file_id]+1)
    #print("Times: ", times)
    t_op = 12
    t_ph = 0.5
    Ts = bc.buildCrownTemp_wPH(times,period_num[file_id],days,substep_num[file_id],t_op,t_ph,Tf=tf_num,dT=td_num)
    temps[file_id] = Ts[1:]
    #print("Temps: ", temps[file_id])

    if td_num == 100: 
        for new_dt in range(0,100,10):
            # retrieving data
            new_temps = temps[file_id][-14*substep_num[file_id]:]  # Use substep_num[file_id]
            dt_vals = new_temps-tf_num  # Use tf_num

            # finding stress at new dt
            closest_idx = np.argmin(np.abs(np.array(dt_vals) - new_dt))
            
            # Get the last 14*substep_num[file_id] values from vmRoutsAll
            last_stresses = elastic_stress_data[file_id]["vmRoutsAll"].iloc[-14*substep_num[file_id]:]
            # Get the stress at the closest index
            new_max_stress = last_stresses.iloc[closest_idx]

            # adding to list
            file_tuples.append((new_dt, tf_num, new_max_stress))

# First get all unique dt and tf values to determine array size
unique_dt = sorted(list(set(t[0] for t in file_tuples)))
unique_tf = sorted(list(set(t[1] for t in file_tuples)))

# Create empty arrays
elastic_stress_grid = np.zeros((len(unique_tf), len(unique_dt)))

# Fill the arrays
for dt, tf, stress in file_tuples:
    dt_idx = unique_dt.index(dt)
    tf_idx = unique_tf.index(tf)
    elastic_stress_grid[tf_idx, dt_idx] = stress


# Used for debugging: plot the elastic stress grid on a 3d surface plot
# Create a meshgrid for the surface plot
# X, Y = np.meshgrid(unique_dt, unique_tf)

# # Create the figure and 3D axes
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Create the surface plot
# surf = ax.plot_surface(X, Y, elastic_stress_grid, 
#                       cmap='viridis',  # You can change the colormap if desired
#                       edgecolor='none',
#                       alpha=0.8)

# # Add a color bar
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Stress (MPa)')

# # Set labels and title
# ax.set_xlabel('DT (°C)')
# ax.set_ylabel('TF (°C)')
# ax.set_zlabel('Stress (MPa)')
# ax.set_title('Elastic Stress Surface')

# # Adjust the viewing angle for better visualization
# ax.view_init(elev=30, azim=45)

# # Show the plot
# plt.show()


predicted_stress_profile = {file_id:[] for file_id in stress_data.keys()}
predicted_damage = {file_id:[] for file_id in stress_data.keys()}
fea_damage = {file_id:[] for file_id in stress_data.keys()}
for file_id in stress_data.keys():
    tf_num = int(file_id.split('TF_')[0])
    td_num = int(file_id.split('TD_')[0].split('_')[1])
    sub = int(file_id.split('_')[2].replace('SUB', ''))
    per = int(file_id.split('_')[3].replace('PER', ''))

    # get residual stress at beginning of 19th day
    residual_stress_value = residual_stress[file_id][-2]
    print(f"\nProcessing {file_id}:")
    
    # get a temp profile
    days = 20
    times = np.linspace(0,days*per,days*per*sub+1)
    t_op = 12
    t_ph = 0.5
    Ts = bc.buildCrownTemp_wPH(times,per,days,sub,t_op,t_ph,Tf=tf_num,dT=td_num)
    temps = Ts[1:]
    temp_profile = temps[-per*sub:]
    
    # Get FEA data for comparison
    fea_stress = stress_data[file_id]["vmRoutsAll"][-per*sub:]
    fea_damage[file_id] = lc.instantaneous_dmg(fea_stress,temp_profile,times[-per*sub:])[2]
    
    for temp in temp_profile:
        # Fix: Use temp-tf_num instead of temp-td_num
        interpolated_stress = interpolate_elastic_stress(tf_num, temp-tf_num, elastic_stress_grid)
        predicted_stress = calculate_actual_stress(interpolated_stress, residual_stress_value, m, b)
        predicted_stress_profile[file_id].append(predicted_stress)

    predicted_damage[file_id] = lc.instantaneous_dmg(predicted_stress_profile[file_id],temp_profile,times[-per*sub:])[2]
    
    print(f"Max FEA stress: {max(fea_stress)}")
    print(f"Max predicted stress: {max(predicted_stress_profile[file_id])}")

#! now instert plot of fea vs prediction
# First Plot: Stress Comparison
plt.figure(figsize=(12, 10))
ax = plt.gca()

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
    # Create small inset axes for stress plot
    stress_inset = inset_axes(ax,
                         width=0.7,  # width of inset
                         height=0.5,  # height of inset
                         loc='center',
                         bbox_to_anchor=(dt, tf, 0.0, 0.0),
                         bbox_transform=ax.transData)
    
    # Get the 20th day stress from FEA
    sub = int(file_id.split('_')[2].replace('SUB', ''))
    per = int(file_id.split('_')[3].replace('PER', ''))
    steps_per_day = sub * per
    
    # Get FEA data
    fea_stress = stress_data[file_id]["vmRoutsAll"][-steps_per_day:]
    
    # Create x-axis points for stress plot
    x_points_stress = np.linspace(0, 1, len(fea_stress))
    
    # Plot FEA and predicted stresses
    stress_inset.plot(x_points_stress, fea_stress, color='blue', linewidth=0.5, label='FEA')
    stress_inset.plot(x_points_stress, predicted_stress_profile[file_id], color='red', linewidth=0.5, label='Predicted')
    
    # Remove ticks and labels
    stress_inset.set_xticks([])
    stress_inset.set_yticks([])
    stress_inset.set_xticklabels([])
    stress_inset.set_yticklabels([])

# Main plot formatting
ax.set_xlabel('DT (Temperature Difference [°C])')
ax.set_ylabel('TF (Fluid Temperature [°C])')
ax.set_title('Stress Comparison')

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
ax.plot(dt_range, tf_line, 'k--', label='y = -2.34x + 791.5')

plt.tight_layout()
plt.show()

# Second Plot: Damage Comparison
plt.figure(figsize=(12, 10))
ax = plt.gca()

# Create main scatter plot with black dots
scatter = ax.scatter(dt_values, tf_values, c='black', s=100)

# Now add inset plots for each point
for dt, tf, file_id in zip(dt_values, tf_values, stress_data.keys()):
    # Create small inset axes for damage plot
    damage_inset = inset_axes(ax,
                         width=0.7,  # width of inset
                         height=0.5,  # height of inset
                         loc='center',
                         bbox_to_anchor=(dt, tf, 0.0, 0.0),
                         bbox_transform=ax.transData)
    
    # Create x-axis points for damage plot
    x_points_damage = np.linspace(0, 1, len(fea_damage[file_id]))
    
    # Plot FEA and predicted damage
    damage_inset.plot(x_points_damage, fea_damage[file_id], color='blue', linewidth=0.5, label='FEA')
    damage_inset.plot(x_points_damage, predicted_damage[file_id], color='red', linewidth=0.5, label='Predicted')
    
    # Remove ticks and labels
    damage_inset.set_xticks([])
    damage_inset.set_yticks([])
    damage_inset.set_xticklabels([])
    damage_inset.set_yticklabels([])

# Main plot formatting
ax.set_xlabel('DT (Temperature Difference [°C])')
ax.set_ylabel('TF (Fluid Temperature [°C])')
ax.set_title('Damage Comparison')

# Create custom legend handles
legend_elements = [
    Line2D([0], [0], color='blue', label='FEA'),
    Line2D([0], [0], color='red', label='Predicted')
]
ax.legend(handles=legend_elements, loc='upper right')

# Add the characteristic line
ax.plot(dt_range, tf_line, 'k--', label='y = -2.34x + 791.5')

plt.tight_layout()
plt.show()

# Third Plot: Error Percentage
plt.figure(figsize=(12, 10))
ax = plt.gca()

# Lists to store all valid error percentages for averaging
valid_errors = []

# Now add error percentages for each point
for dt, tf, file_id in zip(dt_values, tf_values, stress_data.keys()):
    # Calculate percentage error using nansum to ignore NaN values
    fea_damage_sum = np.sum(fea_damage[file_id])
    pred_damage_sum = np.nansum(predicted_damage[file_id])
    
    # Check for zero or negative values
    if fea_damage_sum <= 0:
        percent_error = np.nan
    else:
        percent_error = ((pred_damage_sum - fea_damage_sum) / fea_damage_sum) * 100
        # Only add to valid_errors if it's not NaN
        if not np.isnan(percent_error):
            valid_errors.append(percent_error)
    
    # Determine color based on error percentage
    if np.isnan(percent_error):
        error_color = 'black'
        error_text = 'N/A'
    else:
        if np.abs(percent_error) < 10:
            error_color = 'green'
        elif np.abs(percent_error) < 20:
            error_color = '#FFA500'  # light orange
        elif np.abs(percent_error) < 30:
            error_color = '#FF8C00'  # dark orange
        else:
            error_color = 'red'
        error_text = f'{percent_error:.1f}%'
    
    # Add error percentage text
    ax.text(dt, tf, error_text, 
            color=error_color, 
            ha='center', va='center',
            fontsize=8, fontweight='bold')

# Calculate average error
avg_error = np.mean(np.abs(valid_errors))

# Add statistics box with average error
stats_text = f'Average Absolute Error: {avg_error:.1f}%'
ax.text(0.98, 0.02, stats_text,
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=12,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# Main plot formatting
ax.set_xlabel('DT (Temperature Difference [°C])')
ax.set_ylabel('TF (Fluid Temperature [°C])')
ax.set_title('Error Percentage')

# Add the characteristic line
ax.plot(dt_range, tf_line, 'k--', label='y = -2.34x + 791.5')

plt.tight_layout()
plt.show()






