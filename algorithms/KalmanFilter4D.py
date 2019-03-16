from Matrix import matrix

def filter(x, P):
    for n in range(len(measurements)):
        # prediction
        x = (F * x) + u
        P = F * P * F.transpose()

        # measurement update
        Z = matrix([measurements[n]])
        y = Z.transpose() - (H * x)
        S = H * P * H.transpose() + R
        K = P * H.transpose() * S.inverse()
        x = x + (K * y)
        P = (I - (K * H)) * P

    '''
    x[0] => xCoordinate
    x[1] => yCoordinate
    x[2] => xVelocity
    x[3] => yVelocity
    '''
    print('x= ')
    x.show()
    print('P= ')
    P.show()


########################################

print("### 4-dimensional example ###")

measurements = [[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]]
initial_xy = [4., 12.]

#measurements = [[1., 4.], [6., 0.], [11., -4.], [16., -8.]]
#initial_xy = [-4., 8.]

#measurements = [[1., 17.], [1., 15.], [1., 13.], [1., 11.]]
#initial_xy = [1., 19.]

dt = 0.1

x = matrix([[initial_xy[0]], [initial_xy[1]], [0.], [0.]])  # initial state (location and velocity)
u = matrix([[0.], [0.], [0.], [0.]])  # external motion

#### DO NOT MODIFY ANYTHING ABOVE HERE ####
#### fill this in, remember to use the matrix() function!: ####

# initial uncertainty: 0 for positions x and y, 1000 for the two velocities
P =  matrix([[0., 0., 0., 0.,],
             [0., 0., 0., 0.,],
             [0., 0., 1000., 0.,],
             [0., 0., 0., 1000.,]])
# next state function: generalize the 2d version to 4d
F = matrix([[1., 0., dt, 0.,],
            [0., 1., 0., dt,],
            [0., 0., 1., 0.,],
            [0., 0., 0., 1,]])

# measurement function: reflect the fact that we observe x and y but not the two velocities
H = matrix([[1., 0., 0., 0.,],
            [0., 1., 0., 0.]])

# measurement uncertainty: use 2x2 matrix with 0.1 as main diagonal
R = matrix([[0.1, 0.], [0., 0.1]])

# 4d identity matrix
I = matrix([[1., 0., 0., 0.,],
            [0., 1., 0., 0.,],
            [0., 0., 1., 0.,],
            [0., 0., 0., 1.,]])

filter(x, P)