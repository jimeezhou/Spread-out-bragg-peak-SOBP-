import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import fmin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

##################### read the data #####################
PhysicalDose = pd.read_excel('Data.xlsx', sheet_name=0)
LETRBE = pd.read_excel('Data.xlsx', sheet_name=1)

# 提取列
depth = PhysicalDose.iloc[:, 0].values   # 深度 deep/cm
pristine = PhysicalDose.iloc[:, 1].values  # 原始bragg peak Pristine
##phys_dose = PhysicalDose.iloc[:, 2].values  # PhysicalDose/Gy
##bio_dose = PhysicalDose.iloc[:, 3].values   # BiologyDose

let = LETRBE.iloc[1:, 0].values.astype(float)   # 线性能量传递 LET
rbe = LETRBE.iloc[1:, 1].values.astype(float)   # 相对生物学效应 RBE

######################
step = 0.001

############## bragg 原始峰 pristine 插值
InterDistance = np.arange(0, depth.max() + step, step)

PP = CubicSpline(depth, pristine)  # 样条插值
PristinePP = PP(InterDistance)

########################## Class for Spread out bragg peak ######
class Bragg:
    def __init__(self, Depth, Pristine, LET, RBE, T, step, StartPoint, EndPoint):
        self.Depth = Depth          # 深度 cm
        self.Pristine = Pristine    # 原始bragg peak
        self.LET = LET              # 线性能量传递
        self.RBE = RBE              # 相对生物学效应
        self.T = T                  # 运动时间
        self.step = step
        self.StartPoint = StartPoint
        self.EndPoint = EndPoint

    def Time(self):
        t = np.arange(0, self.T + self.step, self.step)  # 设置总积分区间的时间
        return t

    def Velocity(self, para, t):
        # 速度描述，待拟合
        vt = para[0]*t**4 + para[1]*t**3 + para[2]*t**2 + para[3]*t + para[4]
        # vt = para[0] + para[1]*t
        # vt = para[0]*np.exp(-para[1]*(t-para[2])) + para[3]*t + para[4]
        return vt

    def BraggSpread(self, para):
        t = self.Time()

        RBE_PP = CubicSpline(self.LET, self.RBE)  # RBE样条插值
        RBE_Data = RBE_PP(self.Pristine)

        ################ 积分 ################
        SpreadOut = np.zeros(len(self.Depth))
        Distance = 0.0

        for i in range(len(t)):
            Distance = Distance + self.Velocity(para, self.step * i) * self.step

            ###### Bragg
            # MATLAB: BraggTemp=obj.Pristine(obj.Depth>Distance)
            #         BraggTemp=[BraggTemp; zeros(...)]
            # 左移：截取深度>Distance的值放到开头，末尾补零
            mask = self.Depth > Distance
            n_valid = np.sum(mask)
            BraggTemp = np.zeros(len(self.Pristine))
            BraggTemp[:n_valid] = self.Pristine[mask]

            ############ RBE
            RBETemp = np.zeros(len(RBE_Data))
            RBETemp[:n_valid] = RBE_Data[mask]

            ######## integer (积分)
            SpreadOut = SpreadOut + BraggTemp * RBETemp * self.step

        return SpreadOut

    def PhysicsBraggSpread(self, para):
        t = self.Time()

        ################ 积分 ################
        PhySpreadOut = np.zeros(len(self.Depth))
        Distance = 0.0

        for i in range(len(t)):
            Distance = Distance + self.Velocity(para, self.step * i) * self.step

            ###### Bragg
            # 左移：截取深度>Distance的值放到开头，末尾补零
            mask = self.Depth > Distance
            n_valid = np.sum(mask)
            BraggTemp = np.zeros(len(self.Pristine))
            BraggTemp[:n_valid] = self.Pristine[mask]

            ######## integer (积分)
            PhySpreadOut = PhySpreadOut + BraggTemp * self.step

        return PhySpreadOut

    def Standard_Deviation_of_BraggSpread(self, para):
        SpreadOut = self.BraggSpread(para)
        # S = SpreadOut[(self.Depth > 1) & (self.Depth < 2.2) & (SpreadOut > 1.2)]
        S1 = SpreadOut[(self.Depth > self.StartPoint) & (self.Depth < self.EndPoint)]  # 平台区
        S2 = SpreadOut[self.Depth < self.EndPoint]  # 平台前
        Stddev = np.std(S1) - np.std(S2)  # 平台区要求方差小，平台前区要求方差大，两相减，求极小
        return Stddev


StartPoint = 1
EndPoint = 2.09

bragg = Bragg(InterDistance, PristinePP, let, rbe, 1, step, StartPoint, EndPoint)

para = np.array([-0.2883, 0.1038, 1.3617, 1.7013, -0.0056])
lb = np.array([0, 0, 0, 0])
ub = np.array([0.2, 0.2, 0.2, 0.2]) * 8
for i in range(2):
    # para = fmincon(bragg.Standard_Deviation_of_BraggSpread, para, [], [], [], [], lb, ub)
    para = fmin(bragg.Standard_Deviation_of_BraggSpread, para, disp=False)

print("=" * 50)
print("Optimized para values:")
print(para)
print("=" * 50)

########################## draw figure ##############
# figure(1)
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(InterDistance, PristinePP)
ax1.set_xlabel("Depth/cm")
ax1.legend(["Pristine"])
fig1.savefig('figure1_pristine.png', dpi=150, bbox_inches='tight')
print("Figure 1 saved.")

###########
# figure(2)
fig2, axes = plt.subplots(1, 2, figsize=(16, 5))

# subplot(1,2,1)
axes[0].plot(InterDistance, PristinePP)
axes[0].set_xlabel("Depth/cm")
axes[0].legend(["Pristine"])

# subplot(1,2,2)
axes[1].plot(let, rbe)
axes[1].set_xlabel("LET/Kev")
axes[1].set_ylabel("RBE")

fig2.tight_layout()
fig2.savefig('figure2_subplots.png', dpi=150, bbox_inches='tight')
print("Figure 2 saved.")

####################
# figure(3)
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(InterDistance, bragg.BraggSpread(para), label="Biology Dose")
ax3.plot(InterDistance, bragg.PhysicsBraggSpread(para), label="Physical Dose")
ax3.legend()
fig3.savefig('figure3_bragg_spread.png', dpi=150, bbox_inches='tight')
print("Figure 3 saved.")

########################
# figure(4)
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

# subplot(1,2,1)
t_arr = bragg.Time()
axes4[0].plot(t_arr, bragg.Velocity(para, t_arr))
axes4[0].set_xlabel("Time")
axes4[0].set_ylabel("Velocity")

# subplot(1,2,2)
axes4[1].plot(t_arr, np.cumsum(bragg.Velocity(para, t_arr)) * step)
axes4[1].set_xlabel("Time")
axes4[1].set_ylabel("Distance")

fig4.tight_layout()
fig4.savefig('figure4_velocity_distance.png', dpi=150, bbox_inches='tight')
print("Figure 4 saved.")

print("\nAll figures saved to working directory.")
