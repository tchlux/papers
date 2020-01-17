
from objective import FUNCTIONS, RANDOM, noise, rotate, skew, recenter

# Testing code
from minimize import L_BFGS as minimize
from util.plot import Plot, multiplot

# import timeit
# print("Timing...")
# n = 10000
# print(timeit.timeit("multimin.grad(ones(1000,dtype=float))",number=n,globals=globals())/n)
# exit()

# import fmodpy, os
# ada_mod = fmodpy.fimport(os.path.join("FortranCode","ADA_MOD.f90"),verbose=True)
# help(ada_mod)
# exit()

test_minimize = True
plot_in_2D = True
plot_in_3D = True
add_test_points = False
camera_position = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=-.21),
    eye=dict(x=1.5, y=1.1, z=.8)
)

# FUNCTIONS = FUNCTIONS[3:4]
print([f.__name__ for f in FUNCTIONS])

# Plot each function
for func in FUNCTIONS:
    name = func.__name__
    is_first = (name == FUNCTIONS[0].__name__)
    is_last = (name == FUNCTIONS[-1].__name__)
    # Get the function and plot it 
    name = name.replace("_"," ").title()
    if ("(" in name): name = name[:name.index("(")].strip()

    # func = skew(.5, dimension=2)(func)
    # func = rotate(1, dimension=2)(func)
    # func = noise(1, dimension=1)(func)
    # func = recenter(.5*ones(2), dimension=2)(func)
    print(f"Working on {name}      '{func.__name__}'...")

    if test_minimize:
        from util.system import Timer
        func = FUNCTIONS[1]
        dimension = 6
        budget = 10
        # func = noise(.5, dimension=dimension)(func)
        from numpy import ones
        t = Timer()
        t.start()
        func = recenter(.5*ones(dimension), dimension=dimension)(func)
        func = rotate(1, dimension=dimension)(func)
        t.stop()
        print()
        print("Time-func:", t())
        if (dimension <= 5): print("func.sol: ",func.sol(dimension))
        initial_solution = RANDOM.random(size=(dimension,))
        initial_solution[:] = 1
        if dimension <= 5: print("Start: ",initial_solution)
        t = Timer()
        t.start()
        output = minimize(func, func.grad, initial_solution, budget=budget)
        t.stop()
        from numpy import array
        print("Time-min:", t())
        print("")
        if dimension <= 5: print("Gradient:", func.grad(output[-1]))
        print("Points:  ", array(output).shape)
        from numpy import array
        points = array(output)
        if dimension <= 5:
            print(points[:5])
            print("   ...")
            print(points[-5:])
            print()
        if (dimension <= 2):
            p = Plot("Minimizing some function..")
            p.add_func("Truth", func, *[[-2,2]]*dimension)
            p.add("Steps", *points.T, list(map(func,points)),
                  mode="markers+lines")
            p.show()
        exit()

    if plot_in_2D:
        # Plot the function in 2D
        p = Plot("", f"<i>{name}</i>", "", font_family="Times")
        p.add_func(name, func, [func.lower,func.upper], color=p.color(1))
        g = Plot("", f"<i>{name} derivative</i>", "", font_family="Times")
        g.add_func(f"{name} derivative", func.grad, [-1,1], color=p.color(0))
        multiplot([[p, g]], show_legend=False, file_name=f"{name}-2D.html")

    if plot_in_3D:
        # Plot the function in 3D
        p = Plot(f"{name}","Dimension 1", "Dimension 2", "Function Value",
                 font_family="Times")
        try:
            lower = func.lower
            upper = func.upper
            bounds = list(zip(lower,upper))
        except:
            bounds = [[-1,1],[-1,1]]
        p.add_func(name, func, *bounds, use_gradient=True, plot_points=4000)

        if add_test_points:
            # Add some test points to demonstrate random locations
            for i in range(1000):
                point = func.rand(2)
                value = func(point)
                if (value > 1) or (value < 0): print(*point, value)
                p.add("", *(point[:,None]), [value], color=p.color(0), mode="markers")

        # Plot everything
        p.plot(show_legend=False, file_name=f"{name}-3D.html",
               camera_position=camera_position, z_range=[-.05,1.05]
        )
