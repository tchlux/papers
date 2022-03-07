output_sin = '''
8
 1.958e-04  7.113e-03  2.793e-01
 1.992e-04  6.768e-03  2.653e-01

16
 2.010e-05  1.571e-03  1.338e-01
 2.151e-05  1.652e-03  1.437e-01

32
 2.282e-06  3.688e-04  6.515e-02
 2.520e-06  4.110e-04  7.503e-02

64
 2.721e-07  8.935e-05  3.211e-02
 3.056e-07  1.025e-04  3.837e-02

128
 3.321e-08  2.199e-05  1.593e-02
 3.763e-08  2.561e-05  1.940e-02

256
 4.103e-09  5.455e-06  7.936e-03
 4.670e-09  6.398e-06  9.758e-03

512
 5.099e-10  1.358e-06  3.960e-03
 5.817e-10  1.600e-06  4.894e-03

1024
 6.355e-11  3.389e-07  1.978e-03
 7.258e-11  3.993e-07  2.449e-03

2048
 7.932e-12  8.465e-08  9.887e-04
 9.064e-12  9.984e-08  1.226e-03

4096
 9.908e-13  2.115e-08  4.942e-04
 1.133e-12  2.495e-08  6.124e-04

8192
 1.238e-13  5.287e-09  2.471e-04
 1.417e-13  6.235e-09  3.059e-04

16384
 1.549e-14  1.322e-09  1.235e-04
 1.776e-14  1.559e-09  1.531e-04

32768
 2.109e-15  3.319e-10  6.200e-05
 2.331e-15  3.902e-10  7.662e-05

65536
 8.882e-16  1.251e-10  4.187e-05
 3.331e-16  1.006e-10  4.040e-05

131072
 7.772e-16  1.594e-10  1.075e-04
 3.331e-16  4.442e-11  3.041e-05

262144
 7.772e-16  3.171e-10  3.829e-04
 2.220e-16  7.542e-11  7.259e-05
'''

output_zeros = '''
8
 4.132e-04  1.668e-02  6.693e-01
 3.127e-04  1.668e-02  5.272e-01

16
 4.842e-05  3.843e-03  3.401e-01
 3.364e-05  3.843e-03  2.596e-01

32
 5.850e-06  9.494e-04  1.712e-01
 3.912e-06  9.231e-04  1.288e-01

64
 7.187e-07  2.365e-04  8.582e-02
 4.719e-07  2.263e-04  6.411e-02

128
 8.906e-08  5.903e-05  4.297e-02
 5.796e-08  5.601e-05  3.199e-02

256
 1.108e-08  1.474e-05  2.150e-02
 7.181e-09  1.394e-05  1.598e-02

512
 1.382e-09  3.684e-06  1.075e-02
 8.937e-10  3.475e-06  7.984e-03

1024
 1.726e-10  9.207e-07  5.377e-03
 1.115e-10  8.678e-07  3.991e-03

2048
 2.156e-11  2.301e-07  2.689e-03
 1.392e-11  2.168e-07  1.995e-03

4096
 2.695e-12  5.753e-08  1.344e-03
 1.739e-12  5.419e-08  9.975e-04

8192
 3.375e-13  1.439e-08  6.723e-04
 2.172e-13  1.354e-08  4.987e-04

16384
 4.352e-14  3.609e-09  3.389e-04
 2.753e-14  3.397e-09  2.496e-04

32768
 7.105e-15  9.701e-10  1.796e-04
 3.553e-15  8.553e-10  1.249e-04

65536
 2.665e-15  4.799e-10  1.579e-04
 8.882e-16  2.492e-10  6.457e-05

131072
 2.220e-15  5.600e-10  3.186e-04
 8.882e-16  8.896e-11  5.155e-05

262144
 2.665e-15  1.144e-09  1.038e-03
 8.882e-16  1.666e-10  1.655e-04
'''

# output = [l.strip() for l in output.strip().split("\n")]
# output = [l.strip() for l in output_zeros.strip().split("\n")]
output = [l.strip() for l in output_sin.strip().split("\n")]

print(output)

ns = []
qd0 = []
qd1 = []
qd2 = []
cd0 = []
cd1 = []
cd2 = []

while (len(output) > 0):
    # Pop out empty lines.
    while ((len(output) > 0) and (len(output[0].strip()) == 0)):
        output.pop(0)
    # Extract numeric values for this batch.
    n = int(output.pop(0))
    quintic = list(map(float, output.pop(0).split()))
    cubic = list(map(float, output.pop(0).split()))
    # Add values to lists.
    ns.append(n)
    for (l,v) in zip((qd0,qd1,qd2,cd0,cd1,cd2), quintic+cubic):
        l.append(v)
    # Print out values as sanity check.
    print()
    print("n: ",n)
    print("quintic: ",quintic)
    print("cubic: ",cubic)


from util.plot import Plot


p = Plot("Error approximating e^x on [0,1] with increasing sample points",
         "Number of points", "Max absolute error")

p.add("MQSI d0", ns, qd0, color=0, mode="markers+lines",
      marker_line_width=1,
      line_color="rgba(200,50,50,.35)",
      dash=None, symbol="circle")
p.add("MQSI d1", ns, qd1, color=0, mode="markers+lines",
      marker_line_width=1,
      line_color="rgba(200,50,50,.35)",
      dash="dot", symbol="square")
p.add("MQSI d2", ns, qd2, color=0, mode="markers+lines",
      marker_line_width=1,
      line_color="rgba(200,50,50,.35)",
      dash="dash", symbol="star")
p.add("PCHIP d0", ns, cd0, color=1, mode="markers+lines",
      marker_line_width=1,
      line_color="rgba(50,50,200,.35)",
      dash=None, symbol="circle")
p.add("PCHIP d1", ns, cd1, color=1, mode="markers+lines",
      marker_line_width=1,
      line_color="rgba(50,50,200,.35)",
      dash="dot", symbol="square")
p.add("PCHIP d2", ns, cd2, color=1, mode="markers+lines",
      marker_line_width=1,
      line_color="rgba(50,50,200,.35)",
      dash="dash", symbol="star")

# This is an example of how to control the legend (flat, bottom).
legend = dict(
    xanchor = "center",
    yanchor = "top",
    x = .5,
    y = -.2,
    orientation = "h",
)
# layout_settings = dict(
#     margin = dict(l=60, t=30, b=30),
# )

p.show(axis_settings=dict(type="log"),
       y_axis_settings=dict(tickformat=".0e"),
       legend=legend)
