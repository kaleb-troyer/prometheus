# -*- coding: utf-8 -*-
"""
HALOS main module

This module creates a flux_model object according to user inputs, then creates
an instance of the aimpoint strategy optimization model.

"""
import solve_aim_model
import inputs
import logging
import sys
logging.basicConfig(filename='main_log.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')



if __name__ == "__main__":
    # case_filename = "./../case_inputs/flat_IA_study-220.csv"
    case_filename = "./../case_inputs/flat_50_ca_case.csv"
    logging.info('debug_sess')
    case_name = "first_gurobi"
    filenames = inputs.readCaseFile(case_filename)
    settings = inputs.readSettingsFile(filenames["settings"])
    solve_aim_model.runHourlyCase(case_name, case_name, case_filename, hour_id=settings["hour_idx"], decomp = False, parallel = False)
    # solve_aim_model.runSPHourlyCase(case_name, case_name, case_filename, hour_id=settings["hour_idx"],weather_data = filenames["weather_filename"],sp_aimpoint_heur = True)
