# Load the library and set up the problem data.
library(schumaker)
x = c(0.0, 0.001, 0.02, 0.03)
y = c(0.0, 0.09, 0.1, 0.11)

# Construct the spline fit, rename its evaluator as 'f'.
SchumakerSpline = schumaker::Schumaker(x, y)
f = SchumakerSpline$Spline

# Evaluate at 0.01 to get 0.28, greater than all 'y'.
f(0.01) > max(y)

# Plot the results to see the failed approximation.
pdf(file="test_schumaker.pdf")
plot(x, y, col=1, ylim=c(-0.1,0.35), main="Schumaker", ylab="y", xlab="x")
lines(seq(0,0.03,0.0001), f(seq(0,0.03,0.0001)), col=4)
dev.off()
