from tlux.plot import Plot

xy = list(map(float, open("demo.data", "r").read().split()))
n = int(xy[0])
x = xy[1:n+1]
y = xy[n+1:]
print(n)
print(len(x), x)
print(len(y), y)


points = list(map(float, open("demo.pts", "r").read().split()))
np = int(points[0])
fx = points[1:]
fy = list(map(float, open("demo.out", "r").read().split()))
dfy = list(map(float, open("demo.out_d1", "r").read().split()))
ddfy = list(map(float, open("demo.out_d2", "r").read().split()))
# ify = list(map(float, open("demo.out_i1", "r").read().split()))

print("np: ", np, flush=True)
print("fx: ", len(fx), fx[::len(fx)//11], flush=True)
print("fy: ", len(fy), fy[::len(fy)//11], flush=True)

p = Plot()
p.add("Data", x, y)
p.add("Evaluations", fx, fy, mode="lines")
p.add("Derivative", fx, dfy, mode="lines")
p.add("D2", fx, ddfy, mode="lines")
# p.add("I1", fx, ify, mode="lines")
p.show()
