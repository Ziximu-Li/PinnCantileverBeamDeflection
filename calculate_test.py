import torch
import time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import LBFGS

# 定义物理参数
L = 10.0         # 悬臂梁长度 (m)
E = 210e9       # 弹性模量 (Pa)
I = 1e-7        # 惯性矩 (m^4)
q = 1000        # 均布载荷 (N/m)

maxiter = 10000
# adamsnum = 9500
adamsnum = maxiter
epoch_iter = 1000

torch.manual_seed(1234)  # 设置随机种子以确保实验可重复性
start = time.time()
device = torch.device("cpu")

# 定义一个全连接神经网络类
class PINN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(PINN, self).__init__()
        activation = nn.Softplus  # 使用双曲正切作为激活函数
        # 第一层全连接层，从输入层到隐藏层
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        # 中间隐藏层
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        # 最后一层全连接层，从隐藏层到输出层
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 定义网络的前向传播过程
        x = self.fcs(x)  # 通过第一层全连接层
        x = self.fch(x)  # 通过中间隐藏层
        x = self.fce(x)  # 通过最后一层全连接层
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # 使用Xavier初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 初始化偏置为0

# 定义解析解
def exact_solution(x):
    u = (q * x**2 / (24 * E * I)) * (-L**2 + 2 * L * x - x**2)  # 计算解
    return u

# 计算高阶导数
def compute_derivatives(u, physics, order):
    derivatives = [u]
    for _ in range(order):
        derivatives.append(
            torch.autograd.grad(derivatives[-1], physics, grad_outputs=torch.ones_like(derivatives[-1]),
                                retain_graph=True, create_graph=True)[0]
        )
    return derivatives

def loss_fn(pinn, physics, boundary_left, boundary_right):

    pred = pinn(physics)  # 使用神经网络计算边界点的输出
    pred.requires_grad_(True)
    derivatives = compute_derivatives(pred, physics, 4)
    du_dx, du_dxx, du_dxxx, du_dxxxx = derivatives[1:]

    # 物理方程的损失
    loss_physics = torch.mean((du_dxxxx * E * I + q * torch.ones_like(du_dxxxx))**2 )


    pred_left = pinn(boundary_left)  # 左端的预测值
    pred_right = pinn(boundary_right)  # 右端的预测值

    du_dx_left = torch.autograd.grad(pred_left, boundary_left, torch.ones_like(pred_left), create_graph=True)[0]  # 左端的一阶导数
    du_dx_right = torch.autograd.grad(pred_right, boundary_right, torch.ones_like(pred_right), create_graph=True)[0]  # 右端的一阶导数

    loss_boundary = ((torch.squeeze(pred_left) - 0) ** 2 +
                     (torch.squeeze(pred_right) - 0) ** 2 +
                     (torch.squeeze(du_dx_left) - 0) ** 2 +
                     (torch.squeeze(du_dx_right) - 0) ** 2)  # 计算边界损失的第二部分

    # 最终的loss由两项组成
    lambda_phy = 1e-3
    lambda_boundary = 1e2

    loss = lambda_phy * loss_physics + lambda_boundary * loss_boundary

    return loss

def picture_plot(pinn, test, physics, boundary_left, boundary_right ,epoch, Photo_iter):
    u_exact = exact_solution(test)  # 计算精确解，用于与PINN解进行对比

    # 使用训练好的神经网络 pinn 对测试点 t_test 进行预测，并使用 .detach() 从当前计算图中分离，便于后续处理
    u = pinn(test).detach()
    plt.figure(figsize=(6, 2.5))
    # 在图表上以绿色散点图的形式绘制物理训练点，这些点的 y 值被设置为0，使其在x轴上显示
    plt.scatter(physics.detach()[:, 0],
                torch.zeros_like(physics)[:, 0], s=20, lw=0, color="tab:green", alpha=0.6)
    # 以类似的方式，以红色散点图的形式绘制边界训练点
    plt.scatter(boundary_left.detach()[:, 0], torch.zeros_like(boundary_left)[:, 0], s=20, lw=0, color="tab:red",
                alpha=0.6)
    plt.scatter(boundary_right.detach()[:, 0], torch.zeros_like(boundary_right)[:, 0], s=20, lw=0, color="tab:red",
                alpha=0.6)

    # 绘制精确解（如果可用）作为参考，通常用于比较神经网络的输出和理论上的精确解
    plt.plot(test[:, 0], u_exact[:, 0], label="Exact solution", color="tab:grey", alpha=0.6)
    # 绘制神经网络预测的解决方案，以展示其在测试点上的表现
    plt.plot(test[:, 0], u[:, 0], label="PINN solution", color="tab:green")
    plt.title(f"Training step {epoch}")
    plt.legend()
    plt.savefig(f'Cantilever_{Photo_iter}.png')  # 保存图像
    # plt.show()

def train():
    # 定义一个神经网络用于训练
    pinn = PINN(1, 1, 32, 2)

    # ---------------------- 设置优化求解器 ---------------------- #
    adams_optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    lbfgs_optimiser = torch.optim.LBFGS(pinn.parameters(), lr=1.0, max_iter=1, history_size=10,
                                        line_search_fn="strong_wolfe")
    scheduler = torch.optim.lr_scheduler.StepLR(adams_optimiser, step_size=1000, gamma=0.9)

    def closure():
        lbfgs_optimiser.zero_grad()
        loss = loss_fn(pinn, physics, boundary_left, boundary_right)
        loss.backward()
        return loss

    loss_history = []

    # ---------------------- 设置训练集 ---------------------- #

    # 定义边界点，用于边界损失计算
    boundary_left = torch.tensor(0.).view(-1, 1).requires_grad_(True)
    boundary_right = torch.tensor(10.).view(-1, 1).requires_grad_(True)
    # 创建两个单元素张量，值为0，形状为(1, 1)，需要计算梯度

    # 定义域上的训练点，用于物理损失计算
    physics = torch.linspace(0, 10, 30).view(-1, 1).requires_grad_(True)
    # 创建一个从0到1等间隔的30个点的张量，形状为(30, 1)，需要计算梯度

    test = torch.linspace(0, 10, 300).view(-1, 1)  # 创建一个测试点集，用于可视化

    # ---------------------- 开始训练 ---------------------- #
    adams_start_time = time.time()

    # 训练过程
    Photo_iter = 0  # 图片保存编号

    for epoch in range(maxiter + 1):
        if epoch < adamsnum + 1:
            adams_optimiser.zero_grad()

            loss = loss_fn(pinn, physics, boundary_left, boundary_right)
            # loss.backward()  # 反向传播
            loss.backward(retain_graph=True)    # 反向传播

            nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)   # 梯度裁剪
            adams_optimiser.step()  # 更新网络参数
            scheduler.step()    # 更新学习率

            loss_history.append(loss.item())

            # 每隔1000步绘制和展示训练结果
            if epoch % epoch_iter == 0:
                print("Training step: ", epoch, " ", "loss: ", loss.item())
                picture_plot(pinn, test, physics, boundary_left, boundary_right, epoch, Photo_iter)
                Photo_iter = Photo_iter + 1

            if epoch == adamsnum:
                adams_time = time.time() - adams_start_time
                print(f'Adams time: {adams_time:.2f}s')

        else:
            loss = closure()
            lbfgs_optimiser.step(closure)  # LBFGS优化
            loss_history.append(loss.item())

            if epoch % epoch_iter == 0:
                print("Training step: ", epoch, " ", "loss: ", loss.item())
                picture_plot(pinn, test, physics, boundary_left, boundary_right, epoch, Photo_iter)
                Photo_iter = Photo_iter + 1

    return pinn, test, loss_history

def evaluate(pinn, test, loss_history):
    u_exact = exact_solution(test)  # 计算精确解，用于与PINN解进行对比
    # 使用训练好的神经网络 pinn 对测试点 t_test 进行预测，并使用 .detach() 从当前计算图中分离，便于后续处理
    u = pinn(test).detach()

    # 计算解析解与PINN预测解的相对误差
    epsilon = 1e-2
    relative_error = torch.abs(u - u_exact) / torch.clamp(torch.abs(u_exact), min=epsilon)

    # 绘制相对误差曲线
    plt.figure(figsize=(6, 3))
    plt.plot(test.detach().numpy(), relative_error.detach().numpy(), label="Relative Error")
    plt.title("Relative Error between Exact Solution and PINN Prediction")
    plt.xlabel("x")
    plt.ylabel("Relative Error")
    plt.yscale("log")  # 使用对数刻度以便更好地观察小误差
    plt.legend()
    plt.grid()
    plt.savefig('relative_error_curve.png')  # 保存图像
    plt.show()

    # 绘制损失函数曲线
    plt.figure()
    plt.plot(loss_history[:])
    plt.title('Loss Function Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')  # 可以使用对数坐标
    plt.grid()
    plt.savefig('loss_curve.png')  # 保存图像
    plt.show()  # 显示图像

if __name__ == "__main__":
    pinn, test, loss_history = train()
    evaluate(pinn, test, loss_history)

    timeTotal = time.time() - start
    print(f'Total time: {timeTotal:.2f}s')



