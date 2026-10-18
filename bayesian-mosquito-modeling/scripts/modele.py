import numpy as np


def log_posterior(x, sommes, tailles):
    I = len(sommes)
    z = x[:I]
    mu = x[I]
    log_tau2 = x[I + 1]

    if not np.isfinite(mu) or not np.isfinite(log_tau2):
        return -np.inf

    if not np.all(np.isfinite(z)):
        return -np.inf

    if log_tau2 < -700 or log_tau2 > 700:
        return -np.inf

    if np.max(z) > 700:
        return -np.inf

    tau2 = np.exp(log_tau2)

    out = np.sum(sommes * z - tailles * np.exp(z))
    out = out + np.sum(-0.5 * ((z - mu) ** 2) / tau2 - 0.5 * log_tau2)
    out = out - 0.5 * ((mu - 3.0) ** 2) / 4.0
    out = out + 0.5 * log_tau2 - 0.5 * tau2

    return out


def gradient_log_posterior(x, sommes, tailles):
    I = len(sommes)
    d = I + 2
    z = x[:I]
    mu = x[I]
    log_tau2 = x[I + 1]

    grad = np.zeros(d)

    if not np.isfinite(mu) or not np.isfinite(log_tau2):
        grad[:] = np.nan
        return grad

    if not np.all(np.isfinite(z)):
        grad[:] = np.nan
        return grad

    if log_tau2 < -700 or log_tau2 > 700:
        grad[:] = np.nan
        return grad

    if np.max(z) > 700:
        grad[:] = np.nan
        return grad

    tau2 = np.exp(log_tau2)
    grad[:I] = sommes - tailles * np.exp(z) - (z - mu) / tau2
    grad[I] = np.sum((z - mu) / tau2) - (mu - 3.0) / 4.0
    grad[I + 1] = 0.5 * np.sum(((z - mu) ** 2) / tau2) - 0.5 * I + 0.5 - 0.5 * tau2

    return grad


def hamiltonien(x, p, sommes, tailles, masse):
    log_post = log_posterior(x, sommes, tailles)

    if not np.isfinite(log_post):
        return np.inf

    return -log_post + 0.5 * np.sum((p ** 2) / masse)


def leapfrog(x, p, eps, L, sommes, tailles, masse):
    y = np.copy(x)
    r = np.copy(p)

    for l in range(L):
        grad0 = gradient_log_posterior(y, sommes, tailles)
        r = r + eps * grad0 / 2
        y = y + eps * r / masse

        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(r)):
            return y, -r

        grad1 = gradient_log_posterior(y, sommes, tailles)
        r = r + eps * grad1 / 2

        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(r)):
            return y, -r

    return y, -r


def algorithme_hmc(N, x0, eps, L, sommes, tailles, masse):
    d = len(x0)
    chaine = np.zeros((N, d))
    chaine[0, :] = np.copy(x0)
    accept = np.zeros(N)

    for i in range(1, N):
        p0 = np.sqrt(masse) * np.random.normal(size=d)
        y, r = leapfrog(chaine[i - 1, :], p0, eps, L, sommes, tailles, masse)

        H0 = hamiltonien(chaine[i - 1, :], p0, sommes, tailles, masse)
        H1 = hamiltonien(y, r, sommes, tailles, masse)

        alpha = 0.0
        if np.isfinite(H0) and np.isfinite(H1):
            alpha = min(1.0, np.exp(H0 - H1))

        u = np.random.uniform(0, 1, 1)

        if u < alpha:
            chaine[i, :] = y
            accept[i] = 1
        else:
            chaine[i, :] = chaine[i - 1, :]

    return chaine, accept
