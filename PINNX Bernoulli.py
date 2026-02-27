import braintools
import brainunit as u
import numpy as np

import pinnx

# analytic
def func(x):
    t = x['t']
    y = (1.0 + 3.0 * np.exp(t**2))**(-0.5)   # y(0)=1/2 -> C=3
    return {'y': y}


# time domain
geom = pinnx.geometry.TimeDomain(0.0, 0.5).to_dict_point("t")


# NN
net = pinnx.nn.Model(
    pinnx.nn.DictToArray(t=None),
    pinnx.nn.FNN([1] + [64, 64, 64] + [1], "tanh"),
    pinnx.nn.ArrayToDict(y=None),
)


# Residual（Bernoulli: y' + t*y - t*y^3 = 0）
def ode(x, y):
    # dy/dt
    dy_dt = net.jacobian(x)['y']['t']
    y_val = net(x)['y']
    # Residual：dy_dt + t*y - t*y^3
    return dy_dt + x['t'] * y_val - x['t'] * (y_val ** 3)


# BC y(0)=0.5
ic = pinnx.icbc.IC(lambda x: {'y': 0.5})


# Construct TimePDE problem
data = pinnx.problem.TimePDE(
    geom,
    ode,
    [ic],
    approximator=net,
    num_domain=128,    # residual point taken in the domain
    num_boundary=4,
    solution=func,
    num_test=1000,
    # give two weights：[PDE_weight, IC+BC_weight]
    loss_weights=[1.0, 1.0],
)

# Trainer
trainer = pinnx.Trainer(data)
trainer.compile(braintools.optim.Adam(0.001), metrics=["l2 relative error"])

# Training
trainer.train(iterations=10000)

# Plot
trainer.saveplot(issave=True, isplot=True)

