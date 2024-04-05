from stlpy.systems import LinearSystem


class LinearAffineSystem(LinearSystem):
    """
    A linear affine discrete-time system of the form

    .. math::

        x_{t+1} = A x_t + B u_t + E

        y_t = C x_t + D u_t

    where

        - :math:`x_t \in \mathbb{R}^n` is a system state,
        - :math:`u_t \in \mathbb{R}^m` is a control input,
        - :math:`y_t \in \mathbb{R}^p` is a system output.

    :param A: A ``(n,n)`` numpy array representing the state transition matrix
    :param B: A ``(n,m)`` numpy array representing the control input matrix
    :param E: A ``(n, )`` numpy array representing the affine disturbance vector
    :param C: A ``(p,n)`` numpy array representing the state output matrix
    :param D: A ``(p,m)`` numpy array representing the control output matrix
    """
    def __init__(self, A, B, C, D, E):

        super().__init__(A, B, C, D)

        # Sanity checks on matrix sizes
        assert E.shape == (self.n, ), "E must be an (n,) vector"

        # Store dynamics parameters
        self.E = E

        # Dynamics functions
        self.dynamics_fcn = lambda x, u: A@x + B@u + E
        self.output_fcn = lambda x, u: C@x + D@u