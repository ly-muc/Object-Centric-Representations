import torch


def kl_divergence(mu0, logvar0, mu1, logvar1):

    return -.5 + logvar1 / 2 - logvar0 / 2 - (logvar0.exp() + (mu0
            - mu1) ** 2) / (2 * logvar1.exp())


def infer_k(mu0, logvar0, mu1, logvar1):
    """Function infers the number of pertubations 
    Args:
        mu/sigma (Tensor): Shape: [batch_size, width * height, input_dim]
    """

    b, wh, dim = mu1.shape

    with torch.no_grad():
        # compute divergence
        # shape: [batch_size, width * height]
        delta = torch.mean(kl_divergence(mu0, logvar0, mu1, logvar1), dim=-1)

        # find treshold
        tau = .5 * (torch.max(delta, dim=1).values +
                    torch.min(delta, dim=1).values)  # shape: [batch_size,]

        mask = delta < tau.unsqueeze(1).expand(b, wh)
        k = torch.sum(mask, dim=1)
        mask.unsqueeze(-1).expand(b, wh, dim)

    mu_average = (mu0 + mu1) * .5
    var_average = (logvar0.exp() + logvar1.exp()) * .25

    mu0[mask] = mu_average[mask]
    mu1[mask] = mu_average[mask]

    logvar0[mask] = var_average[mask].log()
    logvar1[mask] = var_average[mask].log()

    return mu0, logvar0, mu1, logvar1, k
