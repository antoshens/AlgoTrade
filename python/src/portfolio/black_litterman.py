import numpy as np

def black_litterman(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    views: np.ndarray | None = None,
    views_transition: np.ndarray | None = None,
    tau: float = .05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the posterior expected returns vector using the Black-Litterman model.

    Combines market equilibrium prior returns (implied from market capitalization/equal weights
    and risk aversion) with subjective investor views and their associated uncertainties.

    Parameters
    ----------
    expected_returns : np.ndarray
        Prior expected annualized returns for the assets (shape: (N,)).
    cov_matrix : np.ndarray
        Annualized covariance matrix of asset returns (shape: (N, N)).
    risk_free_rate : float
        Annualized risk-free rate expressed as a decimal (e.g., 0.045 for 4.5%).
    views : np.ndarray | None, optional
        Investor view vector Q expressing expected returns on specific asset combinations
        (shape: (K, 1) or (K,)), by default None.
    views_transition : np.ndarray | None, optional
        Pick/transition matrix P mapping investor views to assets (shape: (K, N)),
        by default None.
    tau : float, optional
        A scalar representing the degree of uncertainty in the equilibrium vector.
        by default 0.05.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Posterior expected returns vector incorporating the investor views (shape: (N,)).
        - Posterior covariance matrix incorporating the investor views (shape: (N, N)).
    """
    # Prerequisites
    num_assets = cov_matrix.shape[0]
    weights = np.ones(num_assets) / num_assets

    # Risk_aversion matrix
    ret = weights @ expected_returns
    var = weights.T @ cov_matrix @ weights
    '''TODO: make this configurable (if the config for Lambda exists then take the constant value), for robustness'''
    risk_aversion = (ret - risk_free_rate) / var

    # Prior distribution
    pi = (risk_aversion * cov_matrix @ weights).reshape(-1, 1)

    # Investor's views
    if views is not None and views_transition is not None:
        omega = np.diag(np.diag(views_transition @ (tau * cov_matrix) @ views_transition.T)) # matrix of views uncertainty
        inv_omega = np.linalg.inv(omega)
    else:
        views = np.array([[0]])
        views_transition = np.array([[0]])
        inv_omega = np.array([[0]])

    # Posterior vector of returns
    inv_tau_cov = np.linalg.inv(tau * cov_matrix)
    posterior_know = np.array(np.linalg.inv(inv_tau_cov + views_transition.T @ inv_omega @ views_transition))
    prior_know = np.array(inv_tau_cov @ pi + views_transition.T @ inv_omega @ views)
    bl_expected_returns = (posterior_know @ prior_know)

    # Adjusted covariance matrix
    bl_cov_matrix = cov_matrix + posterior_know

    return (bl_expected_returns.flatten(), bl_cov_matrix)
