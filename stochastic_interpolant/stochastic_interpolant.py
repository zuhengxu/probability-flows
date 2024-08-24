import torch
from torchcfm.conditional_flow_matching import pad_t_like_x


class StochasticInterpolator:
    """Base class for StochasticInterpolator. Code adapted from torchcfm.

    This class implements the general stochastic interpolation methods (Eq. (4.1) in [1]), with
    xt = a(t) x0 + b(t) x1 +  gamma(t)* z, 
    where (x0, x1) ~ p(x0, x1), which can be any coupling of p0(x0), p1(x1) , and z ~ N(0, Id) is indepent Gaussian noise.
    Here we denote p1 the target the distribution we want to learn and p0 the base distribution.
    
    There are some additional boundary conditions at t=0, t=1 so that x0 ~ p(x0), x1 ~ p(x1) are repsected: 
    - a(0) = 1, b(0) = 0, 
    - a(1) = 0, b(1) = 1,
    - gamma(0) = 0, gamma(1) = 0.
    

    It implements:
    - Drawing sampling path: xt = a(t) x0 + b(t) x1 +  gamma(t)* z
    - Conditional velocity field used for training, see Eq (2.10) and (2.13) in [1] : ut = dxt/dt = a'(t) x0 + b'(t) x1 + gamma'(t) z

    References
    ----------
    [1] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al. 2023
    """

    def __init__(self, a_t, b_t, gamma_t):
        r"""Initialize the StochasticInterpolator class. 
        It requires a(t), b(t), gamma(t) to define the sample path xt, and their corresponding time derivatives for the conditional velocity.

        Parameters
        ----------
        at : [callable] 
            a(t) function
        bt : [callable] 
            b(t) function
        gamma_t : [callable] 
            gamma(t) function
        """

        self.at = a_t
        self.bt = b_t
        self.gamma_t = gamma_t
        # self.da_dt = da_dt
        # self.db_dt = db_dt
        # self.dgamma_dt = dgamma_dt

    def da_dt(self, t):
        pass

    def db_dt(self, t):
        pass

    def dgamma_dt(self, t):
        pass


    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the sample path mu_t = a(t) x0 + b(t) x1, see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: at * x0 + bt * x0
        """
        t = pad_t_like_x(t, x0)
        return self.at(t) * x0 + self.bt(t) * x1

    def sample_xt(self, x0, x1, t, z):
        """
        Draw a sample from the probability path xt = a(t) x0 + b(t) x1 +  gamma(t)* z, see Eq.(4.1) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        z : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        gamma_t = self.gamma_t(t)
        gamma_t = pad_t_like_x(gamma_t, x0)
        return mu_t + gamma_t * z

    def antithetic_sample_xt(self, x0, x1, t, z):
        """
        Antithetic sampling of the probability path xt = a(t) x0 + b(t) x1 +  gamma(t)* z, see Eq.(6.4) [1].
        This is to address the issue of the variance of the gradient estimator when traininig the velocity b or score directly, 
        but not necessarily when training the denoisers.

        See also https://github.com/interpolants/implicit-interpolants/blob/11479718e1d0d028eeb2d5029ed3c178f789e662/interflow/stochastic_interpolant.py#L128

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        z : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        gamma_t = self.gamma_t(t)
        gamma_t = pad_t_like_x(gamma_t, x0)

        xtp = mu_t + gamma_t * z
        xtn = mu_t - gamma_t * z
        return xtp, xtn


    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def compute_conditional_flow(self, x0, x1, t, z, antithetic=False):
        """
        Compute the conditional vector field ut(x0, x1, z) = a'(t) x0 + b'(t) x1 + gamma'(t) z, see Eq.(2.10) and (2.13) [1].

        The goal is to regress the velocity field vt to the conditional vector field ut(x0, x1, z) using objective Eq. (2.13) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        z : Tensor, shape (bs, *dim)
            represents the noise sample
        t : FloatTensor, shape (bs)
     
        -------
        ut : conditional vector field ut(x0, xt, z) = a'(t) x0 + b'(t) x1 + gamma'(t) z
        """
        da_dt = pad_t_like_x(self.da_dt(t), x0) 
        db_dt = pad_t_like_x(self.db_dt(t), x0)
        dgamma_dt = pad_t_like_x(self.dgamma_dt(t), x0)

        return da_dt * x0 + db_dt * x1 + dgamma_dt * z 


    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt = a(t) x0 + b(t) x1 +  gamma(t)* z, see Eq. (4.1) [1]
        and the conditional vector field ut(x0, x1, z) = a'(t) x0 + b'(t) x1 + gamma'(t) z, see Eq.(2.10) and (2.13) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path xt = a(t) x0 + b(t) x1 +  gamma(t)* z
        ut : conditional vector field ut(x0, xt, z) = a'(t) x0 + b'(t) x1 + gamma'(t) z
        (optionally) z: Tensor, shape (bs, *dim) such that xt = a(t) x0 + b(t) x1 +  gamma(t)* z

        References
        ----------
        [1] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, Albergo et al. 2023
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        z = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, z)
        ut = self.compute_conditional_flow(x0, x1, t, z)
        if return_noise:
            return t, xt, ut, z
        else:
            return t, xt, ut

    def antithetic_sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        z = self.sample_noise_like(x0)
        xtp, xtn = self.antithetic_sample_xt(x0, x1, t, z)
        utp, utn = self.compute_conditional_flow(x0, x1, t, z, anti=True)

        if return_noise:
            return t, (xtp, xtn), (utp, utn), z
        else:
            return t, (xtp, xtn), (utp, utn)

        

class LinearInterpolator(StochasticInterpolator):
    """Child class of StochasticInterpolator. 
    This class implements the linear interpolation method from Eq.(4.9) in [1], with the following specfic interpoloation:
    - a(t) = 1 - t
    - b(t) = t
    - gamma(t) = sqrt(2*t(1-t)) Hence, 
    - a'(t) = -1
    - b'(t) = 1
    - gamma'(t) = (1-2t)/sqrt(2t(1-t))
    
    This class override the a'(t), b'(t), gamma'(t), and compute_conditional_flow methods from the parent class.

    Extracting score function St(x) from trained velocity vt(x):
    ----------------------------------------------------------
    By Eq.(4.12) and (4.13) in [1], we can extract the score function St(x) (for t = 0, 1) from the trained velocity vt(x) as follows:
    In this linear interpolation case, we have:
    - S1(x) = v1(x) + E[x0] + x = v1(x) + x, if E[x0] = 0, e.g., x0 ~ N(0, Id) 
    """
    def __init__(self):
        super(LinearInterpolator, self).__init__(
            a_t=lambda t: 1 - t, 
            b_t=lambda t: t,
            gamma_t=lambda t: torch.sqrt(2*t * (1 - t)),
        )

    # def da_dt(self, t):
    #     return -1.0

    # def db_dt(self, t):
    #     return 1.0

    def dgamma_dt(self, t):
        return (1-2*t)/torch.sqrt(2*t*(1-t))

    def compute_conditional_flow(self, x0, x1, t, z, antithetic=False):
        """
        Compute the conditional vector field ut(x0, x1, z) = a'(t) x0 + b'(t) x1 + gamma'(t) z, using the specific linear interpolation method.

        The goal is to regress the velocity field vt to the conditional vector field ut(x0, x1, z) using objective Eq. (2.13) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        z : Tensor, shape (bs, *dim)
            represents the noise sample
        t : FloatTensor, shape (bs)
     
        -------
        ut : conditional vector field ut(x0, xt, z) = x1 - x0 + gamma'(t) z
        """

        dgamma_dt = pad_t_like_x(self.dgamma_dt(t), x0)
        dIt = x1- x0
        dgamma_z = dgamma_dt * z

        if antithetic:
            return dIt + dgamma_z, dIt - dgamma_z

        return dIt + dgamma_z









if __name__ == "__main__":

    import os
    import matplotlib.pyplot as plt
    import torch
    import torchdiffeq
    from torchvision import datasets, transforms
    from torchvision.transforms import ToPILImage
    from torchvision.utils import make_grid
    from tqdm import tqdm

    from torchcfm.conditional_flow_matching import *
    from torchcfm.models.unet import UNetModel

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 512
    n_epochs = 10

    savedir = "experiments/ball/cond_mnist"
    os.makedirs(savedir, exist_ok=True)
    trainset = datasets.MNIST(
        "experiments/ball/cond_mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = UNetModel(
        dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10, class_cond=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

    SI = LinearInterpolator()

    # training
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            x1 = data[0].to(device)
            y = data[1].to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = SI.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt, y)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")


    # check generated samples
    generated_class_list = torch.arange(10, device=device).repeat(10)
    with torch.no_grad():
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward(t, x, generated_class_list),
            torch.randn(100, 1, 28, 28, device=device),
            torch.linspace(0, 1, 2, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )
    grid = make_grid(
        traj[-1, :100].view([-1, 1, 28, 28]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
    )
    img = ToPILImage()(grid)
    plt.imshow(img)
    plt.show()
