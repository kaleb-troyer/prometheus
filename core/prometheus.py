# Exposes the core set of functions for performing a single aimpoint
# optimization, given the inputs specified under config.inputs.py, or for
# performing a parametric evaluation (with underlying optimizations).
# 2026-03-13
# Kaleb Troyer

import time
import core.losses
import multiprocessing as mp
from core.system import Case, Parameters

class timer():
    """
    Matlab-style process timer.

    ```
    clock = timer()
    clock.tic()
    time.sleep(5)
    t = clock.toc()
    ```
    """
    def __init__(self):
        self.start = 0
        self.bench = 0
        self.timed = 0
    def tic(self):
        if self.timed == 0:
            self.timed = 1
            self.start = time.time()
            print("\nStarting timer...")
        else: print(f"{time.time() - self.start:.4f}")
    def toc(self):
        if self.timed == 1:
            self.timed = 0
            self.bench = time.time()
            print(f"time elapsed: {self.bench-self.start:.4f}")
        else: pass

class Prometheus():

    def __init__(self):

        self.default_params = Parameters.load()
        self.cases = []

    def parametric(self, inputs: dict={}, cores: int=1) -> None:
        """
        Initializes a multithreaded parametric study, where an aimpoint strategy
        is optimized for each individual case. Individual cases are spun out of
        the provided dictionary. 

        Parameters
        ---------------
        inputs : dict
            A dictionary of dictionaries, where each subsequent dictionary is a
            partial implementation of the structures found under config.inputs.
        cores : int
            Total cores allocated to the parametric study and optimizations.
            At least four cores must be allocated for multithreading.
        """

        def combinations(d, results=None) -> list:
            """
            Generates a set of all combinations for the parametric study.

            Parameters
            ---------------
            d : dict
                A dictionary (optionally nested) of lists indicating the
                parametric range of values to be investigated for each key.

            Returns
            ---------------
            results : list
                A list of nested dictionaries of each unique combination.
            """
            if results is None:
                results = [{}]
            for key, value in d.items():
                if isinstance(value, dict):
                    sub_results = combinations(value)
                    results = [
                        {**existing, key: sub}
                        for existing in results
                        for sub in sub_results
                    ]
                else:
                    results = [
                        {**existing, key: val}
                        for existing in results
                        for val in value
                    ]
            return results

        self.studies = combinations(inputs)

        # Each case is created using the _add_case method, after parameters have
        # been updated using the .update method. A single study should take the
        # form of a dictionary of dictionaries, where each subsequent dictionary
        # is a partial implementation of the structures found in config.inputs.
        for study in self.studies:
            params = self.default_params.copy()
            params.update(study)
            self._add_case(params)

        # From here, the function needs to spin up the appropriate number of
        # threads if multithreading, generate a list of jobs from the self.cases
        # list, and assign each job to a thread for optimization. Right now,
        # each case is generated before optimization. However, in the future it
        # will probably make more sense to generate cases from parameters on
        # each individual thread (immediately before optimization) or extend the
        # Case class to generate case features (such as heliostat images) not on
        # .__init__() but at command.

        # as an example:
        if cores <= 3:
            # simple procedure if not multithreading

            print("Initializing parametric study...\n")
            for i, case in enumerate(self.cases):
                self._optimize(case)

            return True

        elif cores >= 4:
            # preparing all jobs and the job manager for multithreading.

            manager = mp.Manager()              # primary multiprocess manager
            parlock = manager.Lock()            # parameter locking tool
            mpqueue = manager.Queue()           # process queue for mplistener
            tracker = manager.Value('i', 0)     # manager variable for process tracking
            cpupool = mp.Pool(processes=cores)  # pool of cpu core resources
            watcher = cpupool.apply_async(      # async mplistener process
                self._mplistener, (mpqueue, tracker, len(self.cases), cores)
            )

            try: # using a try-finally block to ensure resources are always released

                jobs = []
                for case in self.cases:
                    job = cpupool.apply_async(self._mpworker, (case, mpqueue, tracker, parlock))
                    jobs.append(job)
                for job in jobs:
                    job.get()
                mpqueue.put('kill')
                watcher.get()

            finally:

                cpupool.close()
                cpupool.join()
                manager.shutdown()

            return True

    def _mpworker(self, case, queue, tracker, parlock):
        """
        The primary process executed by an individual core during multiprocessed
        parametric evaluations.

        Parameters
        ---------------
        case : Case
            The case for which the aiming strategy is being optimized.
        queue : mp.Manager().queue
            Responsible for sending solutions to the mplistener for processing.
        tracker : mp.Manager().Value
            Tracks the total number of completed jobs.
        parlock : mp.Manager().Lock
            Ensures shared resources are not accessed by multiple threads
            simultaneously.
        """

        # !!! function is unstested and code below is mostly an example of
        # multithread implementation. General process is as follows:
        # 1) read / modify any shared resources using the parlock
        # 2) attempt to set up and perform the optimization; handle failures
        # 3) evaluate results and send to the queue for processing / saving

        # ------- EXAMPLE -------- #
        # The timer can be used to help evaluate time to completion in the
        # mplistener. If thats not needed, freely delete this and the tracker.
        clock = timer(quiet=True)
        clock.tic()

        # If threads must read or modify a shared resource, do so within
        # this parlock block. If they have no such requirement, freely
        # delete this code.
        with parlock:
            pass

        try: # attempting to solve the optimization

            # setup, as an example:
            rate = None # this will probably an imported callable
            time = None # this will probably be an input or Prometheus shared resource
            self._apply_soiling(case, rate, time)

            file = None  # this will probably be an input or Prometheus shared resource
            dims = None  # " "
            start = None # " "
            self._apply_shading(case, file, dims, rate, time, start)

            # finally:
            solution = self._optimize(case)

        except:
            solution = 'Solution not found. Optimization failed to converge.'

        # collecting time elapsed for a single optimization
        elapsed = clock.toc()

        # The tracker is a manager-controlled value that helps evaluate time to
        # completion in the mplistener. If deleting clock, also freely delete this.
        tracker.value += 1

        # Regardless of final implementation, if results are being saved then
        # they must be sent to the queue so that the mplistener can processs them.
        queue.put((study, solution, elapsed))

    def _mplistener(self, queue, tracker, total, cores):
        """
        The secondary process in charge of saving optimization results and
        reporting total progress to the user.

        Parameters
        ---------------
        queue : mp.Manager().queue
            Responsible for sending solutions to the mplistener for processing.
        tracker : mp.Manager().Value
            Tracks the total number of completed jobs.
        total : int
            Total number of jobs.
        cores : int
            Number of cores allocated for the parametric evaluation.
        """

        # This function flushes the terminal buffer and writes a message (which
        # should be structured as a list of strings) to the terminal using the
        # sys.stdout library. The sys library is necessary because, as a
        # separate process, mplistener will not print to the terminal using the
        # default python `print()` function.
        def writer(message):
            for line in message:
                sys.stdout.write(line)
            for _ in range(len(message)-1):
                sys.stdout.write('\033[F')
            sys.stdout.flush()

        # This function evaluates time remaining given the number of completed
        # jobs, the total number of jobs, the amount of time each job took, and
        # the total number of asynchronous processes. It isn't really mission-
        # critical, but can be nice for long parametric studies.
        def timeleft(i, N, times, cores=cores):

            average = np.nanmean(times)
            remaining_iters = N - i
            remaining_time = average * remaining_iters / (cores - 1)

            if remaining_time > 84600:
                message = f'{remaining_time / 84600:.2f} days'
            elif remaining_time > 3600:
                message = f'{remaining_time / 3600:.2f} hours'
            elif remaining_iters > 120:
                message = f'{remaining_time / 60:.2f} minutes'
            else: message = 'nearly complete'

            return message

        # Actual core of the mplistener process begins here:
        while True:

            # mplistener general operation is as follows:
            # 1) continuously watch for results
            try: results = queue.get(timeout=2)
            except Empty:
                continue
            # 2) if the 'kill' message has been sent, end the listener process
            if results=='kill':
                break
            # 3) otherwise, evaluate the contents, process and save results
            elif isinstance(results, tuple) and not isinstance(results[1], str):
                # if the optimization succeeded:
                # 1) unpack results
                # 2) evaluate time remaining (optional)
                # 3) print status to terminal using sys.stdout.message() (optional)
                # 4) write solution to a csv or database
                pass

            else:
                # if the optimization failed:
                # 1) unpack results
                # 2) evaluate time remaining (optional)
                # 3) print status to terminal using sys.stdout.message() (optional)
                # 4) write failure to a csv or database
                pass

    def _optimize(self, case: Case) -> None:
        """
        Wrapper for the optimization, which is imported from Akshay's work.

        Parameters
        ---------------
        case : Case
            The case for which the aiming strategy is being optimized.
        """

        # placeholder function, theoretical usage is as follows:
        # 1) load the optimizer code from core.akshay
        # 2) pass the necessary inputs from the case into the optimizer function
        # 3) after the optimizer finishes, parse output and reassign to the case

        pass

    def _add_case(self, par: Parameters) -> None:
        """
        Creates and appends a new case to the list of Prometheus cases for
        simulation and optimization.

        Parameters
        ---------------
        par : Parameters
            The parameters object, which is used to attempt the creation of the
            corresponding case.
        """

        # TODO: We should have some handling if a case fails to generate.

        try: case = Case(par)
        except:
            return None
        self.cases.append(par)

    def _apply_soiling(self, case: Case, rate: callable, time: float):
        """
        Wrapper for heliostat soiling, imported from core.losses.

        Parameters
        ---------------
        case : Case
            The case that is being soiled.
        rate : callable
            The polynomial f(t) for the rate at which heliostats are soiled.
        time : float
            The time elapsed before heliostat soiling is evaluated.
        """

        # NOTE: Mostly a placeholder function, logic below is an example of
        # potential implementation.

        # need logic for determining soil values based of rate, time, and other
        # variables like when / where / which heliostats are cleaned. for now:
        soiling_value = {key: rate(time) for key, val in case.hel.images.keys()}
        # assumed here the heliostat image dictionary is stored under hel.images
 
        hel_imgs, fluxgrid = core.losses.hel_soiling(
            case.hel.images,
            case.fluxgrid,
            soiling_value
        )

        # and do something with them

        pass

    def _apply_shading(
        self,
        case: Case,
        file: str,
        dims: tuple,
        rate: callable,
        time: float,
        start: tuple,
    ):
        """
        Wrapper for heliostat shading, imported from core.losses.

        Parameters
        ---------------
        case : Case
            The case that is being soiled.
        file : str
            Complete path to the grayscale occlusion image.
        dims : tuple
            A tuple of the x and y dimensions of the occluder, in meters.
        rate : callable
            The function f(t)=(x,y) for the translation of the occluder.
        time : float
            The time elapsed before heliostat soiling is evaluated.
        start : tuple
            A tuple of the x and y starting location of the occluder, in meters.
        """

        # NOTE: Mostly a placeholder function, logic below is an example of
        # potential implementation.

        location = tuple(a + b for a, b in zip(rate(t), start))
        occluder = core.losses.img_to_occl(file)
 
        hel_imgs, fluxgrid = core.losses.hel_shading(
            case.hel.images,
            case.fluxgrid,
            case.layout,
            occluder,
            dims,
            location
        )

        # and do something with them

        pass


if __name__=='__main__':

    pro = Prometheus()

    # example parametric study inputs
    params = {
        'system': {
            't_amb': np.arange(20, 36, 5)
        },
        'receiver': {
            'heat_loss': [20, 25, 30],
        }
    }

    pro.parametric(
        params,
        cores = 4
    )

# EOF
