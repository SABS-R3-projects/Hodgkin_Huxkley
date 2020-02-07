import pints
import Hodgkin_Huxley_class as hh
import Fitzhugh_Nagumo_class as fn
import numpy as np
import matplotlib.pyplot as plt

class SingleOutputInverseProblem:
    def __init__(self, model , times, values):

        self.problem = pints.SingleOutputProblem(model, times, values)
        self.objective_function = pints.SumOfSquaresError(self.problem)
        self.optimiser = pints.CMAES
        self.initial_parameter_uncertainty = None
        self.parameter_boundaries = None

        self.estimated_parameters = None
        self.objective_score = None

    def find_optimal_parameter(self, initial_parameter):
        """Find point in parameter space that optimises the objective function, i.e. find the set of parameters that minimises the
        distance of the model to the data with respect to the objective function.

        Arguments:
            initial_parameter {np.ndarray} -- Starting point in parameter space of the optimisation algorithm.

        Return:
            None
        """
        optimisation = pints.OptimisationController(function=self.objective_function,
                                                    x0=initial_parameter,
                                                    sigma0=self.initial_parameter_uncertainty,
                                                    boundaries=self.parameter_boundaries,
                                                    method=self.optimiser)

        self.estimated_parameters, self.objective_score = optimisation.run()


    def set_objective_function(self, objective_function: pints.ErrorMeasure) -> None:
        """Sets the objective function which is minimised to find the optimal parameter set.

        Arguments:
            objective_function {pints.ErrorMeasure} -- Valid objective functions are [MeanSquaredError,
            RootMeanSquaredError, SumOfSquaresError] in pints.
        """
        valid_obj_func = [pints.MeanSquaredError, pints.RootMeanSquaredError, pints.SumOfSquaresError]

        if objective_function not in valid_obj_func:
            raise ValueError('Objective function is not supported.')

        self.objective_function = objective_function(self.problem)


    def set_optimiser(self, optimiser: pints.Optimiser) -> None:
        """Sets the optimiser to find the "global" minimum of the objective function.

        Arguments:
            optimiser {pints.Optimiser} -- Valid optimisers are [CMAES, NelderMead, PSO, SNES, XNES] in pints.
        """
        valid_optimisers = [pints.CMAES, pints.NelderMead, pints.SNES, pints.XNES]

        if optimiser not in valid_optimisers:
            raise ValueError('Method is not supported.')

        self.optimiser = optimiser


    def set_parameter_boundaries(self, boundaries):
        """Sets the parameter boundaries for inference.

        Arguments:
            boundaries {List} -- List of two lists. [min values, max values]
        """
        min_values, max_values = boundaries[0], boundaries[1]
        self.parameter_boundaries = pints.RectangularBoundaries(min_values, max_values)


model = 2

if model == 1:
    model = hh.Hodgkin_Huxley()
    times = model.time[:1000]
    param = np.array(model.default_params)
    sol = model.simulate(param, times)
    initial_guess = param + 0.1*np.abs(param)*np.random.randn(param.shape[0])

    bounds = [param- np.abs(0.5*param), param + np.abs(0.5*param)]
    print(times)
    problem = SingleOutputInverseProblem(model, times, sol)
    problem.set_parameter_boundaries(bounds)
    problem.find_optimal_parameter(initial_guess)

    print(param)
    print(problem.estimated_parameters)

    sol2 = model.simulate(problem.estimated_parameters, times)

    plt.figure()
    plt.plot(sol[:,0])
    plt.show()
    plt.figure()
    plt.plot(sol2[:,0])
    plt.show()

if model == 2:
    model = fn.Fitzhugh_Nagumo()
    times = np.arange(0,1000,0.1)
    param = np.array(model.default_params)
    sol = model.simulate(param, times)
    initial_guess = param + 0.1 * np.abs(param) * np.random.randn(param.shape[0])

    bounds = [param - np.abs(0.5 * param), param + np.abs(0.5 * param)]
    print(times)
    problem = SingleOutputInverseProblem(model, times, sol)
    problem.set_parameter_boundaries(bounds)
    problem.find_optimal_parameter(initial_guess)

    print(param)
    print(problem.estimated_parameters)

    sol2 = model.simulate(problem.estimated_parameters, times)

    plt.figure()
    plt.plot(sol[:, 0])
    plt.show()
    plt.figure()
    plt.plot(sol2[:, 0])
    plt.show()

