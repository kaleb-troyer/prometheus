# this file serves as an example implementation of prometheus
# at the moment, it is just a placeholder
# 2026-03-01
# Kaleb Troyer

from core.system import Parameters, Case

def main():
    par = Parameters().load()
    des = Case(par)

    des._generate_ideal_flux_grid()
    LTEs = des.get_lifetime_estimate()
    print(LTEs)

    print('\nCase and parameters successfully instantiated!\n')

if __name__=='__main__':

    main()

# EOF
