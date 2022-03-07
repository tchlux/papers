# # Update derivative estimates by using the weighted harmonic mean.
# DO I = 2, ND-1
#    IP1 = I+1
#    IM1 = I-1
#    W1 = 2*X(IP1) - X(I) - X(IM1)
#    W2 = -2*X(IM1) + X(I) + X(IP1)
#    D1 = (Y(IP1) - Y(I)) / (X(IP1) - X(I))
#    D2 = (Y(I) - Y(IM1)) / (X(I) - X(IM1))
#    B = (W1 + W2) / (W1/D2 + W2/D1)
#    A = B * ( 1.0_R8 / D1 - 1.0_R8 / D2 + &
#         W1 / (Y(I) - Y(IM1)) - &
#         W2 / (Y(IP1) - Y(I)) ) / (W1/D2 + W2/D1)
#    IF (FX(I,2) .NE. 0.0_R8) &
#         FX(I,2) = B
#    IF (FX(I,3) .NE. 0.0_R8) &
#         FX(I,3) = 0.0_R8
# END DO
