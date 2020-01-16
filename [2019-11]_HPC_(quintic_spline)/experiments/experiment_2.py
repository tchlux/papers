# Begin program.
import numpy as np
from scipy.stats import norm
# Include the "code" directory into the path and import code custom to this project.
import sys, os
sys.path += [os.path.abspath("code")]
from monotone import monotone_quintic_spline, monotone_cubic_spline
from plot import Plot, multiplot
from system import save, load
from fraction import Fraction
from polynomial import inverse
sys.path.pop(-1)


MAKE_DISTRIBUTION_PLOT = False

# Define the distributions (CDFs) and their weights.
distributions = [
    norm(loc=.2,scale=.05).cdf,
    norm(loc=.45,scale=.08).cdf,
    norm(loc=.85,scale=.03).cdf
]
weights = [.3, .6, .1]
# Define the function that is the CDF for the target distribution.
def dist_cdf(x): return sum(d(x)*w for (d,w) in zip(distributions,weights))

# Approximate the true CDF (in spline format with many points),
# so we can compute its derivative.
approx_dists_file = "e2_approx_distribution.pkl"
try:
    print("Loading distributions..")
    approx_pdf, approx_cdf, approx_inv_cdf = load(approx_dists_file)
except:
    print("Approximating true CDF for differentiation..")
    eval_pts = np.linspace(0,1,10000)
    values = np.array([dist_cdf(x) for x in eval_pts])
    print("  building approximate CDF..")
    approx_cdf = monotone_quintic_spline(eval_pts, values)
    print("  taking derivative of approximate CDF..")
    approx_pdf = approx_cdf.derivative()
    print("  building approximate inverse CDF..")
    approx_inv_cdf = monotone_quintic_spline(values, eval_pts)
    print("  saving all these objects..")
    save((approx_pdf, approx_cdf, approx_inv_cdf), approx_dists_file)

if MAKE_DISTRIBUTION_PLOT:
    # Generate a visual of the CDF.
    print("Building CDF plot..")
    p1 = Plot("","x","P[X ≤ x]", font_family="times", font_size=18)
    p1.add_func("CDF", dist_cdf, [0, 1], color=p1.color(1))

    # Add the PDF.
    print("Building PDF plot..")
    p2 = Plot("","x","P[X ≈ x]", font_family="times", font_size=18)
    p2.add_func("PDF", approx_pdf, [0, 1], color=p2.color(1),
                fill="tozeroy")

    # Plot both together.
    print("Plotting two together..")
    multiplot([[p1],[p2]], show_legend=False, shared_x=True,
              file_name="experiment_2_distribution.html",
              height=200, width=700)


NUM_POINTS = 5
NUM_SAMPLES = 200
NUM_TRIALS = 100
EVAL_POINTS = 500
RANDOM_SEED = 0

results_file = f"e2_results_{NUM_POINTS}_{NUM_SAMPLES}_{NUM_TRIALS}.pkl"
try:
    print("Loading results..")
    cubic, quintic, cubic_results, quintic_results = load(results_file)
    eval_points = np.linspace(0,1, EVAL_POINTS)
except:
    print("Constructing cubic and quintic approximations..")
    # Construct cubic and quintic approximations to the inverse CDF (for sampling).
    x = list(map(Fraction,np.linspace(0,1,NUM_POINTS)))
    y = approx_cdf(x)
    dy = approx_cdf.derivative(1)(x)
    ddy = approx_cdf.derivative(2)(x)
    # Create monotone splines.
    cubic = monotone_cubic_spline(x,values=np.vstack((y,dy)).T)
    quintic = monotone_quintic_spline(x,values=np.vstack((y,dy,ddy)).T)

    # Compute the inverse of these functions (for speeding up other computations).
    print("Inversing these approximations (for generating samples)")
    x = np.linspace(0,1,100)
    inv_cubic = monotone_quintic_spline(cubic(x), x)
    inv_quintic = monotone_quintic_spline(quintic(x), x)

    print("Collecting all data..")    
    np.random.seed(RANDOM_SEED)
    eval_points = np.linspace(0,1, EVAL_POINTS)
    cubic_results   = [list() for i in range(EVAL_POINTS)]
    quintic_results = [list() for i in range(EVAL_POINTS)]
    for t in range(NUM_TRIALS):
        print(f" trial {t}..", end="\r")
        percentiles = np.random.random(size=(NUM_SAMPLES,))
        cubic_vals = inv_cubic(percentiles)
        quintic_vals = inv_quintic(percentiles)
        # Add one step to all the sets of estimated percentiles at each X.
        for i in range(EVAL_POINTS):
            cubic_results[i] += [sum(cubic_vals <= eval_points[i]) / NUM_SAMPLES]
            quintic_results[i] += [sum(quintic_vals <= eval_points[i]) / NUM_SAMPLES]
    # Save the results.
    print("Saving results..")
    save((cubic, quintic, cubic_results, quintic_results), results_file)


print("Making visuals..")
from util.stats.plotting import plot_percentiles

legend = dict(
    xanchor = "center",
    yanchor = "top",
    x = .73,
    y = .5,
    orientation = "v",
    bgcolor="white",
    bordercolor="grey",
    borderwidth=.5
)


p = Plot("","","", font_family="times", font_size=18)
p.add_func("True CDF", dist_cdf, [0, 1], color=p.color((0,0,0)))
plot_percentiles(p, "EDF Values", eval_points, cubic_results, line_width=0)
p.add_func("Cubic CDF", cubic, [0,1], dash="dot", color=p.color(1))
p.show(file_name="experiment_2_cubic_distribution.html", width=400,
       height=365, legend=legend)

p = Plot("","","", font_family="times", font_size=18)
p.add_func("True CDF", dist_cdf, [0, 1], color=p.color((0,0,0)))
p.color_num += 1
plot_percentiles(p, "EDF Values", eval_points, quintic_results, line_width=0)
p.add_func("Quintic CDF", quintic, [0,1], dash="dot", color=p.color(0))
p.show(file_name="experiment_2_quintic_distribution.html", width=400,
       height=365, legend=legend)
