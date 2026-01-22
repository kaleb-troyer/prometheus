Damage_tool 
    • The damage tool class, which is in “damage_tool.py”
    • the material limit class, which is in “material_limits.py”
    • processed FEA data used to extrapolate lifetimes, etc. – dmg_tool_data folder. This folder holds all datasets that the tools can access. The damage tool will not work with this dmg_tool_data subfolder is not present
    • Other scripts used to generate figures in Chapter 3

FEA_data_all
    • Should contain all the relevant FEA raw and processed SRLIFE simulation data from CHTC campaigns.
    • CP# nomenclature indicates that the data originates from Brett Pagel’s work, which used the most up to date scripts and methodology
    • P# nomenclature indicates that the data comes from the initial studies, which had similar methodology and code to the CP# scripts but has slight differences
        ◦ P5 – A230 simulations
        ◦ P7 – 316H simulations
        ◦ P8 – 740H simulations
        ◦ P10 – A282 simulations
        ◦ P11 – A617 simulations
        ◦ P12 – 800H simulations

        ◦ Filename “el” denotes that SRLIFE only considered elastic deformation. If “el” is not present in the filename, it is generally safe to presume that SRLIFE used its “base” deformation model

Real_time_damage_tool  
    • Frankie’s tool. His other files are in “NB_bailey_comparison” subfolder in Chapter_2 folder.
Thermal_and_optical_tools
    • The main folder contains my modified Martinek thermal model scripts, where “jwenn” in the filename indicates I changed something from the original code.
    • The “scripts” subfolder has .py files used to generate figures for chapter 3.
    • In general, the thermal model uses several subfolders:
        ◦ Receivers – contains .json files with receiver information. Used to initialize the model
        ◦ Layouts – contains heliostat layouts for copylot to reference. The receiver .json should have a filestring referencing the specific layout you want copylot to use
        ◦ Aiming – the receiver .json can tell the thermal model to reference a given filestring containing a .csv. The .csv is loaded as a matrix into copylot. Providing a filestring to a .csv stored in ‘aiming’ enables you to use the Solarpilot informed method in the thermal model.
        ◦ Reports – this is where output from the thermal model is typically placed
    • A Copylot instance is also in this folder, but no modifications were made to the publicly accessible version.

Main scripts
The following is a list of scripts that are “main” or “core” files. Note that this list should be used as a starting point rather than an exhaustive list.
    • Test_damage_tool_all_materials – good main file for learning how to use the damage tool. Contains sample run code.
    • Make_ideal_flux_profiles – good intro to making ideal fluxmaps/grids for receivers, but a better version of this file exists in Chapter 4.
    • run_steady_state_baseline_study or run_steady_state_optimized_study  contains all main code required to solve a full thermal model.  Commented out lines of code represent various post-processing options for plotting, generating reports, etc.
