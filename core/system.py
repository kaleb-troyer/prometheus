
import sys

sys.path.append('product/config/')

from inputs import *
from schema import *
from loader import *

class System():

    # TASKS
    # - generate a field from user inputs
    # - generate a set of flux images from inputs
    # - identify case-driving inputs

    # ORDER OF OPERATIONS
    # - select case-driving inputs
    # - generate hash from those inputs
    # - generate cases dir from hash
    # - store hash/case information in db
    # - generate field layout if none exists
    # - generate flux images if none exists

    def __init__(self):

        self.rec = load(Receiver, RECEIVER)
        self.hel = load(Heliostats, HELIOSTATS)

    def generate_field_from_inputs(self):

        # Example in Jacob's code?

        pass



if __name__=='__main__':

    sys = System()
    print(sys.rec)

# EOF
