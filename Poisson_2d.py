# ==========================================
# 1. JAX MONKEY-PATCH (必须放在最前面)
# 解决 jax.tree_map 弃用导致的 AttributeError
# ==========================================
import jax
from jax import tree_util
if not hasattr(jax, 'tree_map'):
    jax.tree_map = tree_util.tree_map

# ==========================================
# 2. 标准库与依赖导入
# ==========================================
import braintools
import brainunit as u
import numpy as np
from jax.experimental.sparse import COO
from scipy.special import gamma
import pinnx
import matplotlib.pyplot as plt

alpha = 1.8

# ==========================================
# 3. PDE 与 网络定义
# ==========================================
def fpde(x, y, int_mat):
    r"""
    \int_theta D_theta^alpha u(x)
    """
    y = y['y']
    x = pinnx.utils.dict_to_array(x)
    if isinstance(int_mat, (list, tuple)) and len(int_mat) == 3:
        rowcols = np.asarray(int_mat[0], dtype=np.int32).T
        data = int_mat[1]
        ini_mat = COO((data, rowcols[0], rowcols[1]), shape=int_mat[2])
        lhs = ini_mat @ y
    else:
        lhs = u.math.matmul(int_mat, y)
        
    lhs *= gamma((1 - alpha) / 2) * gamma((2 + alpha) / 2) / (2 * np.pi ** 1.5)
    x = x[: len(lhs)]
    
    # 使用 keepdims=True 避免维度广播错误 (N,1) - (N,) -> (N,N)
    rhs = (
        2 ** alpha
        * gamma(2 + alpha / 2)
        * gamma(1 + alpha / 2)
        * (1 - (1 + alpha / 2) * u.math.sum(x ** 2, axis=1, keepdims=True))
    )
    return lhs - rhs

net = pinnx.nn.Model(
    pinnx.nn.DictToArray(x1=None, x2=None),
    pinnx.nn.FNN([2] + [20] * 4 + [1], "tanh", braintools.init.KaimingUniform(),
                 output_transform=lambda x, y: (1 - u.math.sum(x ** 2, axis=1, keepdims=True)) * y),
    pinnx.nn.ArrayToDict(y=None),
)

def func(x):
    x = pinnx.utils.dict_to_array(x)
    y = (u.math.abs(1 - u.linalg.norm(x, axis=1, keepdims=True) ** 2)) ** (1 + alpha / 2)
    return {'y': y}

# ==========================================
# 4. 几何修复与问题初始化
# ==========================================
class PatchedDisk(pinnx.geometry.Disk):
    def inside(self, x):
        # 将 dynamic mesh 产生的 numpy 数组转换为 brainunit 张量，避免底层报错
        return super().inside(u.math.asarray(x))

geom = PatchedDisk([0, 0], 1).to_dict_point('x1', 'x2')
bc = pinnx.icbc.DirichletBC(func)

data = pinnx.problem.FPDE(
    geom,
    fpde,
    alpha,
    bc,
    [8, 100],
    net,
    meshtype='dynamic',
    num_domain=100,
    num_boundary=1,
    solution=func
)

# ==========================================
# 5. 模型训练
# ==========================================
model = pinnx.Trainer(data)
model.compile(braintools.optim.Adam(1e-3)).train(iterations=20000)
# 关闭原有的 2D 散点图，防止弹窗打断后面的 3D 绘图
model.saveplot(issave=True, isplot=False) 

# ==========================================
# 6. 预测与评估
# ==========================================
X = geom.random_points(1000)
y_true = func(X)
y_pred = model.predict(X)
print("L2 relative error:", pinnx.metrics.l2_relative_error(y_true, y_pred))

# ==========================================
# 7. 3D 曲面绘图
# ==========================================
# 重新生成较密集的点阵用于画平滑的曲面
X_plot = geom.random_points(5000)
y_true_dict = func(X_plot)
y_pred_dict = model.predict(X_plot)

# 提取坐标
X_arr = np.array(pinnx.utils.dict_to_array(X_plot))
x1 = X_arr[:, 0]
x2 = X_arr[:, 1]

# 提取并展平 z 值
z_true = np.array(y_true_dict['y']).flatten()
z_pred = np.array(y_pred_dict['y']).flatten()
z_err = np.abs(z_true - z_pred)

# 创建 1x3 的子图：真实值、预测值、误差
fig = plt.figure(figsize=(18, 5))

# 图 1：真实解
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_trisurf(x1, x2, z_true, cmap='viridis', edgecolor='none')
ax1.set_title('True Solution', fontsize=14)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('u')
fig.colorbar(surf1, ax1=ax1, shrink=0.5, aspect=10, pad=0.1)

# 图 2：预测解
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_trisurf(x1, x2, z_pred, cmap='plasma', edgecolor='none')
ax2.set_title('PINN Prediction', fontsize=14)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('u')
fig.colorbar(surf2, ax2=ax2, shrink=0.5, aspect=10, pad=0.1)

# 图 3：绝对误差
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_trisurf(x1, x2, z_err, cmap='coolwarm', edgecolor='none')
ax3.set_title('Absolute Error', fontsize=14)
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('|True - Pred|')
fig.colorbar(surf3, ax3=ax3, shrink=0.5, aspect=10, pad=0.1)

plt.tight_layout()
plt.show()