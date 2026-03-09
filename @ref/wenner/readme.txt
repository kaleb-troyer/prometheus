This is a top level description of the file structure and where to find key scripts.
In general, files have been organized so that the scripts and data are placed in a chapter folder according to the figure or discussion’s location in the dissertation.
Chapter_2 
    • source code for conducting FEA campaigns via the CHTC and making the damage maps.
    • Frankie’s code used for comparison of the NB method to our FEA method (Section 2.6 in dissertation. All of Frankie’s code is here, but it is also available from OneDrive (ask Mike). 
Chapter_3 
    • Most up to date version of the damage tool
    • FEA campaign data, both processed and unprocessed
    • Frankie’s real time damage tool
    • Modified thermal model that uses copylot instead of old API
    • Lightweight-configured thermal model
    • Updated copylot version
Chapter_4
    • Most up to date aiming informer module
    • Local version of HALOS
    • Damage informed aiming heuristic
    • Informed SPT aiming heuristic (generate aiming files via aiming informer)
    • LCOH analysis scripts and functions
PHiL-microgrid (see dissertation appendix)
    • Procedures for running microgrid
    • PV inverter control code
    • Data and post-processing scripts used for appendix paper’s chapters 2 and 3.1. For 3.2’s scripts and data, see Erik Haag’s work (ask Mike).
submission 
    • administrative paperwork for dissertation submission 
    • complete dissertation, as submitted to graduate school for publishing
sendable_dissertation
    • dissertation broken into main (sem appendix) and appendix (sem appendix’s appendix) for reduced file size. Helps if sending via outlook as attachment.
Requirements_HALOSnew_env.txt – this is a pseudo requirements.txt for the main python environment I used on my SEL-39 windows computer. Note that this environment differs from what I used on the Linux WS01 computer.
