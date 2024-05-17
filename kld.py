import pdb
import numpy as np

def c_approx(kappa: float, beta: float) -> float:
    """Corrected approximation for the normalization constant c(kappa, beta)."""
    return 2 * np.pi * np.exp(kappa) * ((kappa - 2 * beta) * (kappa + 2 * beta))**(-0.5)

def del_kappa(kappa: float, beta: float) -> float:
    """Calculates the first derivative wrt kappa.

    Args:
        kappa: A numerical value representing the variable kappa.
        beta: A numerical value representing the variable beta.

    Returns:
        The calculated result of the expression.
    """

    # Handle potential division by zero
    if kappa - 2 * beta == 0 or kappa + 2 * beta == 0:
        raise ZeroDivisionError("Division by zero occurred in the expression.")

    # Factor out a negative sign from the numerator
    numerator = -2 * np.pi * (4 * beta**2 + kappa - kappa**2) * np.exp(kappa)  

    # Simplify the denominator using the difference of squares
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2)

    result = numerator / denominator
    return result

def del_2_kappa(kappa: float, beta: float) -> float:
    """Calculates the second derivative wrt kappa.

    Args:
        kappa: A numerical value representing the variable kappa.
        beta: A numerical value representing the variable beta.

    Returns:
        The calculated result of the expression.
    """

    # Handle potential division by zero
    if kappa - 2 * beta == 0 or kappa + 2 * beta == 0:
        raise ZeroDivisionError("Division by zero occurred in the expression.")

    # Calculate the numerator
    numerator = 2 * np.pi * (kappa**4 - 2 * kappa**3 + (2 - 8 * beta**2) * kappa**2 + 8 * beta**2 * kappa + 16 * beta**4 + 4 * beta**2) * np.exp(kappa)

    # Calculate the denominator
    denominator = (kappa - 2 * beta)**(5/2) * (kappa + 2 * beta)**(5/2)

    result = numerator / denominator
    return result

def del_beta(kappa, beta):
  """Calculates the third npematical expression with inputs kappa and beta.

  Args:
      kappa: A numerical value representing the variable kappa.
      beta: A numerical value representing the variable beta.

  Returns:
      The calculated result of the expression.
  """

  # Handle potential division by zero
  if kappa - 2 * beta == 0 or 2 * beta + kappa == 0:
    raise ZeroDivisionError("Division by zero occurred in the expression.")

  # Calculate the numerator
  numerator = 8 * np.pi * np.exp(kappa) * beta

  # Calculate the denominator using the difference of squares pattern
  denominator = (kappa - 2 * beta) * (kappa + 2 * beta) ** (3/2)

  result = numerator / denominator
  return result

def E_x(Q_matrix: np.ndarray, kappa: float, beta: float) -> np.ndarray:
    """Calculates the expected value of the elastic strain in the x-direction (E_x) for a transversely isotropic material.

    This function uses the stiffness matrix (`Q_matrix`), the shape factor (`kappa`), and the fiber orientation angle (`beta`) to compute the expected value of E_x. It relies on the closed-form approximations for the compliance matrix (`c_approx`) and its derivative with respect to kappa (`del_kappa`).

    Args:
        Q_matrix: A 3x3 NumPy array representing the stiffness matrix of the transversely isotropic material.
        kappa: The shape factor of the inclusion.
        beta: The fiber orientation angle (in radians).

    Returns:
        A 1D NumPy array of length 3 representing the expected value of the elastic strain in the x-direction.
    """

    return del_kappa(kappa, beta) / c_approx(kappa, beta) * Q_matrix[0]

def ExxT(
    Q_matrix: np.ndarray, kappa: float, beta: float
) -> np.ndarray:
    """Calculates the expected value of the outer product of the elastic strain tensor (ExxT) for a transversely isotropic material.

    This function uses the stiffness matrix (`Q_matrix`), the shape factor (`kappa`), and the fiber orientation angle (`beta`) to compute the expected value of ExxT. It leverages the closed-form approximations for the derivatives of the compliance matrix with respect to kappa and beta.

    Args:
        Q_matrix: A 3x3 NumPy array representing the stiffness matrix of the transversely isotropic material.
        kappa: The shape factor of the inclusion.
        beta: The fiber orientation angle (in radians).

    Returns:
        A 3x3 NumPy array representing the expected value of the outer product of the elastic strain tensor (ExxT).
    """
    
    # Helper functions (presumably defined elsewhere) to calculate compliance matrix derivatives
    c = c_approx(kappa, beta)
    c_k = del_kappa(kappa, beta)
    c_kk = del_2_kappa(kappa, beta)
    c_beta = del_beta(kappa, beta)

    # Calculate the lambda values
    lambda_1 = c_k / c
    lambda_2 = (c - c_kk + c_beta) / (2 * c)
    lambda_3 = (c - c_kk - c_beta) / (2 * c)

    # Construct the lambda matrix
    lambda_matrix = np.array([[lambda_1, 0, 0], [0, lambda_2, 0], [0, 0, lambda_3]])

    # Calculate and return ExxT
    return Q_matrix @ lambda_matrix @ Q_matrix.T

def kld(
    kappa_a: float, beta_a: float, Q_matrix_a: np.ndarray,
    kappa_b: float, beta_b: float, Q_matrix_b: np.ndarray
) -> float:
    """Calculates the expected log-likelihood ratio between two Kent distributions.

    This function assumes the expectations under both Kent distributions have been pre-calculated.
    The rotation matrices are passed as arrays, and their components are extracted within the function.

    Args:
        Ex_a: Expected value of x under Kent distribution 'a' (1D NumPy array).
        ExxT_a: Expected value of x * x.T under Kent distribution 'a' (2D NumPy array).
        ca: Normalization constant for Kent distribution 'a'.
        kappa_a, beta_a: Concentration and ovalness parameters for Kent distribution 'a'.
        Q_matrix_a: Rotation matrix for Kent distribution 'a' (3x3 NumPy array).
        Ex_b, ExxT_b, cb, kappa_b, beta_b, Q_matrix_b: Same as above, but for Kent distribution 'b'.

    Returns:
        The expected log-likelihood ratio (float).
    """
    gamma_a1, gamma_a2, gamma_a3 = Q_matrix_a[:, 0], Q_matrix_a[:, 1], Q_matrix_a[:, 2] # Extract the first and second column vectors of Gamma_a
    gamma_b1, gamma_b2, gamma_b3 = Q_matrix_b[:, 0], Q_matrix_b[:, 1], Q_matrix_a[:, 2] # Extract the first and second column vectors of Gamma_b

    ca = c_approx(kappa_a, beta_a)
    cb = c_approx(kappa_b, beta_b)

    ExxT_a = ExxT(Q_matrix_a, kappa_a, beta_a)
    ExxT_b = ExxT(Q_matrix_b, kappa_b, beta_b)

    Ex_a = E_x(Q_matrix_a, kappa_a, beta_a)
    Ex_b = E_x(Q_matrix_b, kappa_b, beta_b)

    result = (
        np.log(cb / ca) 
        + (kappa_a * gamma_a1.T - kappa_b * gamma_b1.T) @ Ex_a 
        + (beta_a * gamma_a2.T @ ExxT_a @ gamma_a2 
        - beta_b * gamma_b2.T @ ExxT_b @ gamma_b2)
        + (beta_a * gamma_a3.T @ ExxT_a @ gamma_a3 
        - beta_b * gamma_b3.T @ ExxT_b @ gamma_b3)
    )

    return result.item() 

if __name__ == "__main__":
    
    #first distribution
    kappa_a = 10
    beta_a = 1
    Q_matrix_a = np.array([[1,0,0], [0,1,0],[0,0,1]])

    #Second distribution
    kappa_b = 5
    beta_b = 1
    Q_matrix_b = np.array([[1,0,0], [0,1,0],[0,0,1]])

    kld_value = kld(kappa_a, beta_a, Q_matrix_a, kappa_b, beta_b, Q_matrix_b)

    print(f"KLD values is {kld_value}")

    #pdb.set_trace()


