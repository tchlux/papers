# This code is used to update the tables throughout the paper.

del_prep = []
del_app = []
vor_prep = []

# Forest fires:  0.0150505304337
print("Forest fires:")
sq_time = 0.0150505304337
prep_time = 0.0000249 + sq_time
app_time = 0.0385 - sq_time
total_time = 1.93 - (sq_time*50)
del_prep += [prep_time]
del_app += [app_time]
vor_prep += [0.000182+sq_time]
print(f"Delaunay  {prep_time:.2e}  {app_time:.2e}  {total_time:.2e}")
print(f"Voronoi   {vor_prep[-1]:.2e}")
print()


# Parkinson's:   2.46990501881
print("Parkinson's:")
sq_time = 2.46990501881
prep_time = 0.000190 + sq_time
app_time = 3.69 - sq_time
total_time = 2170 - (sq_time*587)
del_prep += [prep_time]
del_app += [app_time]
vor_prep += [0.298+sq_time]
print(f"Delaunay  {prep_time:.2e}  {app_time:.2e}  {total_time:.2e}")
print(f"Voronoi   {vor_prep[-1]:.2e}")
print()

# Weather:       0.664243936539
print("Weather:")
sq_time = 0.664243936539
prep_time = 0.000116 + sq_time
app_time = 1.55 - sq_time
total_time = 403 - (sq_time*260)
del_prep += [prep_time]
del_app += [app_time]
vor_prep += [0.0112+sq_time]
print(f" Delaunay  {prep_time:.2e}  {app_time:.2e}  {total_time:.2e}")
print(f" Voronoi   {vor_prep[-1]:.2e}")
print()

# Credit cards:  3.11890363693
print("Credit cards:")
sq_time = 3.11890363693
prep_time = 0.000243 + sq_time
app_time = 6.83 - sq_time
total_time = 3800 - (sq_time*556)
del_prep += [prep_time]
del_app += [app_time]
vor_prep += [0.197+sq_time]
print(f" Delaunay  {prep_time:.2e}  {app_time:.2e}  {total_time:.2e}")
print(f" Voronoi   {vor_prep[-1]:.2e}")
print()

# I/O zone:      0.34315931797
print("I/O Zone:")
sq_time = 0.34315931797
prep_time = 0.000517 + sq_time
app_time = 0.353 - sq_time
total_time =  106. - (sq_time*301)
del_prep += [prep_time]
del_app += [app_time]
vor_prep += [0.0173+sq_time]
print(f" Delaunay  {prep_time:.2e}  {app_time:.2e}  {total_time:.2e}")
print(f" Voronoi   {vor_prep[-1]:.2e}")
print()


# --------------------------------------------------------------------

# Compute the new averages.
print("Averages:")
print(f" Delaunay  {sum(del_prep)/len(del_prep):.2e}  {sum(del_app)/len(del_app):.2e}")
print(f" Voronoi   {sum(vor_prep)/len(vor_prep):.2e}")
