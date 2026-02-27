import braintools
import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
import pinnx

def pde(x, y):
    hessian = net.hessian(x)
    dy_xx = hessian["y"]["x"]["x"]
    return -dy_xx - u.math.pi ** 2 * u.math.sin(u.math.pi * x['x'])

def func(x):
    return {'y': u.math.sin(u.math.pi * x['x'])}

# 1. Define the network
net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x=None),
    pinnx.nn.FNN([1] + [50] * 3 + [1], "tanh"),
    pinnx.nn.ArrayToDict(y=None),
)

geom = pinnx.geometry.Interval(-1, 1).to_dict_point('x')
bc = pinnx.icbc.DirichletBC(func)

# 2. Initial data setup: intentionally use fewer domain points (16)
data = pinnx.problem.PDE(
    geom, pde, bc, net, num_domain=16, num_boundary=2, solution=func, num_test=100
)

# 3. Initial training
trainer = pinnx.Trainer(data)
trainer.compile(braintools.optim.Adam(0.001), metrics=["l2 relative error"])
print("--- Starting initial training ---")
trainer.train(iterations=10000)


# ==========================================
# 4. Start RAR (Residual-based Adaptive Refinement) loop
# ==========================================
print("\n--- Starting RAR adaptive sampling ---")

X_search = geom.uniform_points(1000, boundary=False)

max_err = 1.0
max_rar_loops = 10  # Slightly increase the maximum number of loops
iteration = 0
threshold = 0.005   # Note: Threshold is now based on the [maximum residual]

while max_err > threshold and iteration < max_rar_loops:
    y_pred = trainer.predict(X_search)
    residual = pde(X_search, y_pred)
    
    err_eq = np.abs(np.asarray(residual)) 
    
    # [Core modification]: Look at the maximum error, not the mean error
    max_err = np.max(err_eq)
    mean_err = np.mean(err_eq)
    print(f"RAR Loop {iteration + 1}: Max residual = {max_err:.3e} (Mean residual = {mean_err:.3e})")

    if max_err <= threshold:
        print("Max residual threshold reached, stopping adaptive sampling.")
        break
        
    # [Optimization]: Find the indices of the top 3 points with the maximum residual at once
    num_new_points = 3
    top_k_indices = np.argsort(err_eq.flatten())[-num_new_points:]
    
    # Extract the coordinates of these points
    new_points_x = X_search['x'][top_k_indices]
    print(f"-> Adding new collocation points (x): {new_points_x}")
    
    # Add these new points to the training data
    new_anchors = {'x': np.array(new_points_x)}
    data.add_anchors(new_anchors)
    
    # Retrain to digest these new points
    print("-> Retraining for 3000 steps...")
    trainer.train(iterations=3000) 
    
    iteration += 1

# ==========================================
# 5. Result visualization
# ==========================================
print("\n--- Training completed, generating plots ---")
trainer.saveplot(issave=True, isplot=True)

# Plot the final PDE residual to verify the effect
x_plot = geom.uniform_points(1000, True)
y_plot = pde(x_plot, trainer.predict(x_plot))
plt.figure()
plt.plot(x_plot['x'], np.asarray(y_plot))
plt.xlabel("x")
plt.ylabel("PDE residual")
plt.title("PDE Residual after RAR")
plt.show()