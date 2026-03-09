import math
import matplotlib.pyplot as plt
import numpy as np
# from sympy import symbols, log, simplify

# Parameters

sigma_eq = 383e6 #PA
E = 170e9 #PA
t_stab = 1*60*60  #s
n = 6.6 #[-]
A = 2.688e-45 #1/Pa^6.6-s)
m = 0
Q = 322 #kj/mol
R = 8.314 /1000 #KJ/mol-K
T = 630 +273.15 #K
def norton_bailey_gonzalez(sigma_eq, t_stab, E, T, R = 8.314/1000, n =6.6, m = 0.00, A = 2.6880e-45, Q = 322):
    sigma_eq_r = sigma_eq-E*( (1-n)*( ((sigma_eq/E)**(1-n) )/(1-n) - A*(E**n)*math.exp(-Q/(R*T))* (t_stab**(m+1))/(m+1) ) )**(1/(1-n))
    return sigma_eq_r

def get_Temp_Norton_Bailey_Gonzalez(sigma_eq, sigma_eq_r, t_stab, E, R = 8.314/1000, n =6.6, m = 0.00, A = 2.6880e-45, Q = 322):
    try:
        C5 = ((sigma_eq/E)**(1-n))/(1-n)
        C4 = (sigma_eq-sigma_eq_r)/E
        C3 = 1/(1-n)*C4**(1-n)-C5
        C2 = -(A*E**n)*(t_stab**(m+1)/(m+1))
        C1 = C3/C2
        if C1 <= 0:  # log of negative or zero is undefined
            return float('nan')
        T = -Q/R*1/np.log(C1)
        return T
    except:
        return float('nan')

# #! Relaxed Stress as a function of time
# sigma_eq_r_funct_t = []
# for t_stab in range(1,100,1):
#     sigma_eq_r = sigma_eq-E*( (1-n)*( ((sigma_eq/E)**(1-n) )/(1-n) - A*(E**n)*math.exp(-Q/(R*T))* (t_stab**(m+1))/(m+1) ) )**(1/(1-n))
#     sigma_eq_r_funct_t.append(sigma_eq_r)

# plt.figure(figsize=(10, 6))
# plt.plot(sigma_eq_r_funct_t)
# plt.title('Relaxed Stress as a function of time')
# plt.xlabel('Time')
# plt.ylabel('Relaxed Stress')
# plt.show()

# #! Relaxed Stress as a function of Temperature
# sigma_eq_r_funct_temp = []
# temperatures = np.linspace(200+273, 600+273, 100)
# stabilization_times = [1*24*60*60, 25*24*60*60, 50*24*60*60, 75*24*60*60, 100*24*60*60]  # 5 different times

# plt.figure(figsize=(10, 6))

# for t_stab in stabilization_times:
#     temp_curve = []  # Store results for this specific t_stab
#     for temp in temperatures:
#         sigma_eq_r = sigma_eq-E*( (1-n)*( ((sigma_eq/E)**(1-n) )/(1-n) - A*(E**n)*math.exp(-Q/(R*temp))* (t_stab**(m+1))/(m+1) ) )**(1/(1-n))
#         temp_curve.append(sigma_eq_r)
#     plt.plot(temperatures, temp_curve, label=f't_stab = {t_stab}h')

# plt.title('Relaxed Stress as a function of Temperature')
# plt.xlabel('Temperature (K)')
# plt.ylabel('Relaxed Stress (Pa)')
# plt.legend()
# plt.show()

    

# # Define the variables
# sigma, sigma_relax, time, E, R, n, m, A, Q = symbols('sigma sigma_relax time E R n m A Q')

# # Define the first equation (T)
# T_eq_maple = Q / (R * (-log(-((-sigma + sigma_relax) * ((sigma - sigma_relax) / E)**(-n)
#                          + sigma * (sigma / E)**(-n)) * (m + 1) / (A * (-1 + n)))
#                     + (m + 1) * log(time) + (n + 1) * log(E)))

# # Define the second equation based on the function logic
# C5 = ((sigma / E)**(1 - n)) / (1 - n)
# C4 = (sigma - sigma_relax) / E
# C3 = 1 / (1 - n) * C4**(1 - n) - C5
# C2 = -(A * E**n) * (time**(m + 1) / (m + 1))
# C1 = C3 / C2
# T_eq_original = -Q / R * 1 / log(C1)

# # Simplify the difference
# are_equal = simplify(T_eq_maple - T_eq_original) == 0
# print("Are the equations the same?", are_equal)

# #! Numerical verification
# def calculate_T_maple(sigma_val, sigma_relax_val, time_val, E_val, R_val, n_val, m_val, A_val, Q_val):
#     try:
#         return float(Q_val / (R_val * (-np.log(-((-sigma_val + sigma_relax_val) * 
#                    ((sigma_val - sigma_relax_val) / E_val)**(-n_val) + 
#                    sigma_val * (sigma_val / E_val)**(-n_val)) * 
#                    (m_val + 1) / (A_val * (-1 + n_val))) + 
#                    (m_val + 1) * np.log(time_val) + (n_val + 1) * np.log(E_val))))
#     except:
#         return float('nan')

# def calculate_T_original(sigma_val, sigma_relax_val, time_val, E_val, R_val, n_val, m_val, A_val, Q_val):
#     try:
#         c5 = ((sigma_val / E_val)**(1 - n_val)) / (1 - n_val)
#         c4 = (sigma_val - sigma_relax_val) / E_val
#         c3 = 1 / (1 - n_val) * c4**(1 - n_val) - c5
#         c2 = -(A_val * E_val**n_val) * (time_val**(m_val + 1) / (m_val + 1))
#         c1 = c3 / c2
#         return float(-Q_val / R_val * 1 / np.log(c1))
#     except:
#         return float('nan')

# # Test with random values
# num_tests = 10000000 # 10 million tests
# max_diff = 0
# total_valid = 0
# all_diffs = []  # Store all differences

# print("\nTesting random cases:")
# print("sigma_val, sigma_relax_val, T_maple, T_original, Difference")
# print("-" * 70)

# for _ in range(num_tests):
#     # Generate random values within reasonable ranges
#     sigma_val = np.random.uniform(100e6, 500e6)  # 100-500 MPa
#     sigma_relax_val = np.random.uniform(1e6, sigma_val)  # Less than sigma
#     time_val = np.random.uniform(1*60*60, 100*60*60)  # 1-100 hours in seconds
#     E_val = 170e9  # Keep constant as it's a material property
#     R_val = 8.314/1000  # Gas constant
#     n_val = 6.6  # Keep constant as it's a material property
#     m_val = 0.0  # Keep constant as per original
#     A_val = 2.688e-45  # Keep constant as it's a material property
#     Q_val = 322  # Keep constant as it's a material property

#     T_maple = calculate_T_maple(sigma_val, sigma_relax_val, time_val, E_val, R_val, n_val, m_val, A_val, Q_val)
#     T_original = calculate_T_original(sigma_val, sigma_relax_val, time_val, E_val, R_val, n_val, m_val, A_val, Q_val)
    
#     if not (np.isnan(T_maple) or np.isnan(T_original)):
#         diff = abs(T_maple - T_original)
#         max_diff = max(max_diff, diff)
#         total_valid += 1
#         all_diffs.append(diff)  # Store all differences
        
#         if diff > 1e-6:  # Only print cases with significant differences
#             print(f"{sigma_val/1e6:.1f}MPa, {sigma_relax_val/1e6:.1f}MPa, {T_maple:.2f}K, {T_original:.2f}K, {diff:.2e}")

# print("\nSummary:")
# print(f"Total valid tests: {total_valid}/{num_tests}")
# print(f"Maximum difference: {max_diff:.2e}")
# print(f"Average difference: {np.mean(all_diffs):.2e}")
# print(f"Equations are {'effectively equal' if max_diff < 1e-10 else 'different'}")

# #! Statistical analysis of the differences
# def analyze_differences(diffs):
#     diffs = np.array(diffs)
#     print("\nStatistical Analysis:")
#     print(f"Mean difference: {np.mean(diffs):.2e}")
#     print(f"Median difference: {np.median(diffs):.2e}")
#     print(f"Standard deviation: {np.std(diffs):.2e}")
#     print(f"95th percentile: {np.percentile(diffs, 95):.2e}")
#     print(f"99th percentile: {np.percentile(diffs, 99):.2e}")
    
#     # Create histogram of differences
#     plt.figure(figsize=(10, 6))
#     plt.hist(diffs, bins=50)
#     plt.yscale('log')
#     plt.xlabel('Absolute Difference')
#     plt.ylabel('Count (log scale)')
#     plt.title('Distribution of Differences Between Equations')
#     plt.show()

# analyze_differences(all_diffs)
