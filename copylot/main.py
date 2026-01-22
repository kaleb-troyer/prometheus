
from parameters import designParameters
from copylot import CoPylot

import os

def test(threads=2):

    cp = CoPylot()

    # dp = designParameters(cp)

    # defining design parameters
    # path = os.path.join(os.getcwd(), 'copylot', 'climates')
    # file = 'USA CA Daggett (TMY2).csv'
    # des_par_amb = dp.get(category='amb')
    # des_par_amb['weather_file'] = os.path.join(path, file)
    # dp.update(des_par_amb)

    # setting up instance, assigning parameters
    rm = cp.data_create()
    # dp.assign_to_instance(rm)

    # Executing SolarPILOT Simulation
    assert cp.generate_layout(rm, nthreads=threads)
    assert cp.simulate(rm, nthreads=threads)
    summary = cp.summary_results(rm, save_dict=True)
    results = params.get() | summary
    assert cp.data_free(rm)

    return results

if __name__=='__main__':

    test()

#EOF
