import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import fmin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import os
import time
import io
import streamlit.components.v1 as components
import tempfile

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Spread Out Bragg Peak",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B3A5C;
        margin-bottom: 0.1rem;
    }
    .wechat-tag {
        font-size: 1.05rem;
        font-weight: 600;
        color: #2E86C1;
        margin-bottom: 0.8rem;
        padding: 4px 10px;
        background: #EBF5FB;
        border-radius: 6px;
        display: inline-block;
    }
    .sidebar-section {
        margin-top: 0.45rem;
        margin-bottom: 0.3rem;
        padding-left: 10px;
        border-left: 3px solid #1B3A5C;
    }
    .sidebar-section h4 {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1B3A5C;
        border-bottom: 2px solid #1B3A5C;
        padding-bottom: 4px;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .result-box {
        background: #f8fbff;
        border: 1px solid #d0e0f0;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    div[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Bragg Class (from original code)
# ============================================================
class Bragg:
    def __init__(self, Depth, Pristine, LET, RBE, T, step, StartPoint, EndPoint):
        self.Depth = Depth
        self.Pristine = Pristine
        self.LET = LET
        self.RBE = RBE
        self.T = T
        self.step = step
        self.StartPoint = StartPoint
        self.EndPoint = EndPoint

    def Time(self):
        t = np.arange(0, self.T + self.step, self.step)
        return t

    def Velocity(self, para, t):
        vt = para[0]*t**4 + para[1]*t**3 + para[2]*t**2 + para[3]*t + para[4]
        return vt

    def BraggSpread(self, para):
        t = self.Time()
        RBE_PP = CubicSpline(self.LET, self.RBE)
        RBE_Data = RBE_PP(self.Pristine)
        SpreadOut = np.zeros(len(self.Depth))
        Distance = 0.0
        for i in range(len(t)):
            Distance = Distance + self.Velocity(para, self.step * i) * self.step
            mask = self.Depth > Distance
            n_valid = np.sum(mask)
            BraggTemp = np.zeros(len(self.Pristine))
            BraggTemp[:n_valid] = self.Pristine[mask]
            RBETemp = np.zeros(len(RBE_Data))
            RBETemp[:n_valid] = RBE_Data[mask]
            SpreadOut = SpreadOut + BraggTemp * RBETemp * self.step
        return SpreadOut

    def PhysicsBraggSpread(self, para):
        t = self.Time()
        PhySpreadOut = np.zeros(len(self.Depth))
        Distance = 0.0
        for i in range(len(t)):
            Distance = Distance + self.Velocity(para, self.step * i) * self.step
            mask = self.Depth > Distance
            n_valid = np.sum(mask)
            BraggTemp = np.zeros(len(self.Pristine))
            BraggTemp[:n_valid] = self.Pristine[mask]
            PhySpreadOut = PhySpreadOut + BraggTemp * self.step
        return PhySpreadOut

    def Standard_Deviation_of_BraggSpread(self, para):
        SpreadOut = self.BraggSpread(para)
        S1 = SpreadOut[(self.Depth > self.StartPoint) & (self.Depth < self.EndPoint)]
        S2 = SpreadOut[self.Depth < self.EndPoint]
        Stddev = np.std(S1) - np.std(S2)
        return Stddev

# ============================================================
# Load demo data
# ============================================================
@st.cache_data
def load_demo_data():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    bragg_path = os.path.join(demo_dir, 'demo_bragg.json')
    rbe_path = os.path.join(demo_dir, 'demo_rbe.json')

    with open(bragg_path, 'r') as f:
        bragg_data = json.load(f)
    bp_df = pd.DataFrame(bragg_data['data'], columns=bragg_data['columns'])

    with open(rbe_path, 'r') as f:
        rbe_data = json.load(f)
    rbe_df = pd.DataFrame(rbe_data['data'], columns=rbe_data['columns'])

    return bp_df, rbe_df

demo_bragg_df, demo_rbe_df = load_demo_data()

# ============================================================
# Session State Init
# ============================================================
if 'bragg_df' not in st.session_state:
    st.session_state.bragg_df = demo_bragg_df.copy()
if 'rbe_df' not in st.session_state:
    st.session_state.rbe_df = demo_rbe_df.copy()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = True
if 'results' not in st.session_state:
    st.session_state.results = None
if 'switch_to_opt' not in st.session_state:
    st.session_state.switch_to_opt = False

# ============================================================
# Title
# ============================================================
# Sidebar - Title + Navigation + Controls
# ============================================================
with st.sidebar:
    st.markdown('<div class="main-title">Spread Out Bragg Peak (SOBP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="wechat-tag">WeChat: icecoler</div>', unsafe_allow_html=True)

    # --- Part 1: Data Loading ---
    st.markdown('<div class="sidebar-section"><b>📊 1. Data Loading</b></div>', unsafe_allow_html=True)

    # Data source toggle
    data_source = st.radio(
        "Data Source",
        ["Demo Data", "Upload Files"],
        index=0,
        key="data_source_radio",
        horizontal=True,
    )

    if data_source == "Upload Files":
        bragg_file = st.file_uploader(
            "Upload BraggPeak",
            type=["xlsx", "xls", "csv"],
            key="bragg_uploader",
            help="Excel/CSV with columns: depth, Pristine",
        )
        rbe_file = st.file_uploader(
            "Upload RBE (Relative Biological Effectiveness)",
            type=["xlsx", "xls", "csv"],
            key="rbe_uploader",
            help="Excel/CSV with columns: LET, RBE",
        )

        if bragg_file is not None and rbe_file is not None:
            try:
                if bragg_file.name.endswith('.csv'):
                    bp_df = pd.read_csv(bragg_file)
                else:
                    bp_df = pd.read_excel(bragg_file)

                if rbe_file.name.endswith('.csv'):
                    rbe_df = pd.read_csv(rbe_file)
                else:
                    rbe_df = pd.read_excel(rbe_file)

                st.session_state.bragg_df = bp_df
                st.session_state.rbe_df = rbe_df
                st.session_state.data_loaded = True
                st.success("Files loaded successfully!")
            except Exception as e:
                st.error(f"Error loading files: {e}")
                st.session_state.data_loaded = False
        else:
            st.info("Please upload both BraggPeak and RBE files.")
            st.session_state.data_loaded = False
    else:
        st.session_state.bragg_df = demo_bragg_df.copy()
        st.session_state.rbe_df = demo_rbe_df.copy()
        st.session_state.data_loaded = True
        st.info("Using built-in demo data.")

    # --- Part 1 continued: Interpolation Step ---
    st.markdown('<div class="sidebar-section"><b>⚙️ Interpolation Step</b></div>', unsafe_allow_html=True)
    step = st.number_input(
        "Step (cm)",
        min_value=0.0001,
        max_value=0.1,
        value=0.001,
        step=0.0001,
        format="%.4f",
        help="Interpolation step for densifying discrete data. Smaller = finer resolution but slower.",
    )

    # --- Part 2: Spread Width ---
    st.markdown('<div class="sidebar-section"><b>📐 2. Spread Width</b></div>', unsafe_allow_html=True)

    bp_df = st.session_state.bragg_df
    depth_min = float(bp_df.iloc[:, 0].min())
    depth_max = float(bp_df.iloc[:, 0].max())

    col_sp1, col_sp2 = st.columns(2)
    with col_sp1:
        start_point = st.number_input(
            "StartPoint (cm)",
            min_value=0.0,
            max_value=depth_max,
            value=1.0,
            step=0.01,
            format="%.2f",
        )
    with col_sp2:
        end_point = st.number_input(
            "EndPoint (cm)",
            min_value=start_point + 0.01,
            max_value=depth_max,
            value=2.09,
            step=0.01,
            format="%.2f",
        )

    # --- Part 3: Run ---
    st.markdown('<div class="sidebar-section"><b>🚀 3. Run Optimization</b></div>', unsafe_allow_html=True)
    n_iter = st.number_input(
        "Optimization Iterations",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        help="Number of fmin optimization rounds. More = better fit but slower.",
    )

    run_btn = st.button("▶ Run Optimization", type="primary", use_container_width=True)
    if run_btn:
        st.session_state.switch_to_opt = True

# ============================================================
# Main Content Area
# ============================================================

if not st.session_state.data_loaded:
    st.warning("Please load data from the sidebar to begin.")
    st.stop()

bp_df = st.session_state.bragg_df
rbe_df = st.session_state.rbe_df

# Parse BraggPeak data
depth = bp_df.iloc[:, 0].values.astype(float)
pristine = bp_df.iloc[:, 1].values.astype(float)

# Parse RBE data (skip header row if present)
if rbe_df.iloc[0, 0] in ['Kev/um', 'LET', 'LET/Kev']:
    rbe_clean = rbe_df.iloc[1:].astype(float)
else:
    rbe_clean = rbe_df.astype(float)
let = rbe_clean.iloc[:, 0].values
rbe = rbe_clean.iloc[:, 1].values

# Interpolation
InterDistance = np.arange(0, depth.max() + step, step)
PP = CubicSpline(depth, pristine)
PristinePP = PP(InterDistance)

# ============================================================
# Tabbed Content: Data Preview / Optimization / Simulation / Help
# ============================================================
tab_preview, tab_opt, tab_sim, tab_help = st.tabs([
    "📊 Data Preview", "🚀 Optimization Results", "🎬 Simulation", "📖 Help"
])

# Auto-switch to Optimization Results tab when sidebar Run is clicked
if st.session_state.switch_to_opt:
    st.session_state.switch_to_opt = False
    components.html(
        """<script>
        var tabs = window.parent.document.querySelectorAll('[data-testid="stTabs"] button');
        if (tabs.length >= 2) { tabs[1].click(); }
        </script>""",
        height=0,
    )

# ---- Tab 1: Data Preview ----
with tab_preview:
    # Data preview content

    col_data1, col_data2 = st.columns(2)

    with col_data1:
        st.markdown("**BraggPeak Data**")
        fig_bp, ax_bp = plt.subplots(figsize=(6, 3.5))
        ax_bp.plot(depth, pristine, 'b-', linewidth=1.5)
        ax_bp.set_xlabel("Depth / cm", fontsize=10)
        ax_bp.set_ylabel("Pristine", fontsize=10)
        ax_bp.set_title("Bragg Peak", fontsize=11, fontweight='bold')
        ax_bp.grid(True, alpha=0.3)
        fig_bp.tight_layout()
        st.pyplot(fig_bp)
        plt.close(fig_bp)

        with st.expander("View BraggPeak Table"):
            st.dataframe(bp_df, use_container_width=True, height=200)

    with col_data2:
        st.markdown("**RBE Data (Relative Biological Effectiveness)**")
        fig_rbe, ax_rbe = plt.subplots(figsize=(6, 3.5))
        ax_rbe.plot(let, rbe, 'r-', linewidth=1.5)
        ax_rbe.set_xlabel("LET / Kev·μm⁻¹", fontsize=10)
        ax_rbe.set_ylabel("RBE", fontsize=10)
        ax_rbe.set_title("LET vs RBE", fontsize=11, fontweight='bold')
        ax_rbe.grid(True, alpha=0.3)
        fig_rbe.tight_layout()
        st.pyplot(fig_rbe)
        plt.close(fig_rbe)

        with st.expander("View RBE Table"):
            st.dataframe(rbe_df, use_container_width=True, height=200)

    # Interpolation info
    st.markdown(
        f"**Interpolation:** Step = {step} cm → "
        f"Original {len(depth)} points → Interpolated {len(InterDistance)} points  |  "
        f"Depth range: [{depth.min():.2f}, {depth.max():.2f}] cm"
    )

    # Spread Width section
    # Spread Width
    st.markdown(
        f"Platform region: **{start_point:.2f} cm** → **{end_point:.2f} cm**  "
        f"(width = **{end_point - start_point:.2f} cm**)"
    )

    # Highlight the spread region on the pristine curve
    fig_spread_region, ax_sr = plt.subplots(figsize=(8, 3.5))
    ax_sr.plot(InterDistance, PristinePP, 'b-', linewidth=1.5, label='Pristine')
    ax_sr.axvspan(start_point, end_point, alpha=0.2, color='orange', label=f'Spread Region [{start_point:.2f}, {end_point:.2f}]')
    ax_sr.axvline(start_point, color='orange', linestyle='--', linewidth=1)
    ax_sr.axvline(end_point, color='orange', linestyle='--', linewidth=1)
    ax_sr.set_xlabel("Depth / cm", fontsize=10)
    ax_sr.set_ylabel("Pristine", fontsize=10)
    ax_sr.set_title("Pristine Peak with Spread Region", fontsize=11, fontweight='bold')
    ax_sr.legend(fontsize=9)
    ax_sr.grid(True, alpha=0.3)
    fig_spread_region.tight_layout()
    st.pyplot(fig_spread_region)
    plt.close(fig_spread_region)

# ---- Tab 1: Optimization Results ----
with tab_opt:
    # Optimization results content
    st.markdown("""
    <style>
        div[data-testid="stButton"][data-key="run_btn_tab"] > button {
            width: auto !important;
            min-width: unset !important;
            display: inline-block !important;
            margin-left: 0 !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
    run_btn_tab = st.button("▶ Run Optimization", type="primary", key="run_btn_tab")

    if run_btn or run_btn_tab:
        with st.spinner("Optimizing SOBP parameters... This may take a moment."):
            progress = st.progress(0, text="Initializing optimization...")

            bragg = Bragg(InterDistance, PristinePP, let, rbe, 1, step, start_point, end_point)

            para = np.array([-0.2883, 0.1038, 1.3617, 1.7013, -0.0056])

            for i in range(n_iter):
                progress.progress(int((i + 1) / n_iter * 60), text=f"Optimization round {i+1}/{n_iter}...")
                para = fmin(bragg.Standard_Deviation_of_BraggSpread, para, disp=False)

            progress.progress(70, text="Computing spread profiles...")

            bio_spread = bragg.BraggSpread(para)
            phys_spread = bragg.PhysicsBraggSpread(para)

            progress.progress(100, text="Done!")
            time.sleep(0.3)
            progress.empty()

            # Store results
            st.session_state.results = {
                'para': para.tolist(),
                'bio_spread': bio_spread.tolist(),
                'phys_spread': phys_spread.tolist(),
                'start_point': start_point,
                'end_point': end_point,
                'step': step,
            }

    # Display results if available
    if st.session_state.results is not None:
        res = st.session_state.results
        para = np.array(res['para'])
        bio_spread = np.array(res['bio_spread'])
        phys_spread = np.array(res['phys_spread'])
        sp = res['start_point']
        ep = res['end_point']

        # Optimized parameters
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("**Optimized Velocity Parameters:**")
        para_cols = st.columns(5)
        labels = ['a₀ (t⁴)', 'a₁ (t³)', 'a₂ (t²)', 'a₃ (t)', 'a₄ (const)']
        for j, (label, val) in enumerate(zip(labels, para)):
            with para_cols[j]:
                st.metric(label, f"{val:.6f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Figure 3: SOBP curves
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(InterDistance, bio_spread, 'r-', linewidth=2, label="Biology Dose (SOBP)")
        ax3.plot(InterDistance, phys_spread, 'b-', linewidth=2, label="Physical Dose (SOBP)")
        ax3.axvspan(sp, ep, alpha=0.15, color='orange', label=f'Spread Region [{sp:.2f}, {ep:.2f}]')
        ax3.axvline(sp, color='orange', linestyle='--', linewidth=1)
        ax3.axvline(ep, color='orange', linestyle='--', linewidth=1)
        ax3.set_xlabel("Depth / cm", fontsize=11)
        ax3.set_ylabel("Dose", fontsize=11)
        ax3.set_title("Spread Out Bragg Peak", fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # Figure 4: Velocity & Distance
        bragg = Bragg(InterDistance, PristinePP, let, rbe, 1, res['step'], sp, ep)
        t_arr = bragg.Time()

        fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4.5))

        axes4[0].plot(t_arr, bragg.Velocity(para, t_arr), 'g-', linewidth=1.5)
        axes4[0].set_xlabel("Time", fontsize=10)
        axes4[0].set_ylabel("Velocity", fontsize=10)
        axes4[0].set_title("Velocity vs Time", fontsize=11, fontweight='bold')
        axes4[0].grid(True, alpha=0.3)

        axes4[1].plot(t_arr, np.cumsum(bragg.Velocity(para, t_arr)) * res['step'], 'm-', linewidth=1.5)
        axes4[1].set_xlabel("Time", fontsize=10)
        axes4[1].set_ylabel("Distance", fontsize=10)
        axes4[1].set_title("Distance vs Time", fontsize=11, fontweight='bold')
        axes4[1].grid(True, alpha=0.3)

        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # Combined Figure: Left=Distance vs Time, Right=3D Annular Pie
        fig_combined = plt.figure(figsize=(18, 8))

        # Left: Quarter-Circle Distance vs Time (0 ~ 2π, 2 cycles)
        ax5 = fig_combined.add_subplot(121)

        d_actual = np.cumsum(bragg.Velocity(para, t_arr)) * res['step']
        D_max = float(np.max(d_actual))

        n_pts = 4000
        x = np.linspace(0, 2 * np.pi, n_pts)
        half_cycle = np.pi

        d_qc = np.zeros(n_pts)
        for i in range(2):
            x0 = i * half_cycle
            mask = (x >= x0) & (x < x0 + half_cycle)
            tau = (x[mask] - x0) / half_cycle

            d_segment = np.zeros(int(np.sum(mask)))
            rise = tau <= 0.5
            fall = tau > 0.5

            s_r = tau[rise] * 2
            theta_r = 3 * np.pi / 2 - s_r * np.pi / 2
            d_segment[rise] = D_max + D_max * np.sin(theta_r)

            s_f = (tau[fall] - 0.5) * 2
            theta_f = np.pi + s_f * np.pi / 2
            d_segment[fall] = D_max + D_max * np.sin(theta_f)

            d_qc[mask] = d_segment

        ax5.plot(x, d_qc, 'r-', linewidth=2, label='Quarter-Circle Distance (Ideal)')
        ax5.plot(t_arr * 2 * np.pi, d_actual, 'b--', linewidth=1.5, alpha=0.7, label='Optimized Distance')

        for i in range(1, 2):
            ax5.axvline(i * half_cycle, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        for i in range(2):
            ax5.text((i + 0.5) * half_cycle, -0.06 * D_max,
                     f'Cycle {i+1}', ha='center', fontsize=9, color='gray')

        ax5.set_xlabel("Time (0 → 2π)", fontsize=11)
        ax5.set_ylabel("Distance / cm", fontsize=11)
        ax5.set_title("Quarter-Circle Distance vs Time (2 cycles in 2π)", fontsize=13, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)

        # Right: 3D Annular Pie
        ax6 = fig_combined.add_subplot(122, projection='3d')

        r_inner = 5.0
        r_outer = 10.0

        h_qc = d_qc.copy()

        n_theta_3d = 200
        n_r_3d = 10
        x_3d = np.linspace(0, 2 * np.pi, n_theta_3d)

        d_qc_3d = np.zeros(n_theta_3d)
        for i in range(2):
            x0 = i * np.pi
            mask = (x_3d >= x0) & (x_3d < x0 + np.pi)
            tau = (x_3d[mask] - x0) / np.pi
            d_seg = np.zeros(int(np.sum(mask)))
            rise = tau <= 0.5
            fall = tau > 0.5
            s_r = tau[rise] * 2
            theta_r = 3 * np.pi / 2 - s_r * np.pi / 2
            d_seg[rise] = D_max + D_max * np.sin(theta_r)
            s_f = (tau[fall] - 0.5) * 2
            theta_f = np.pi + s_f * np.pi / 2
            d_seg[fall] = D_max + D_max * np.sin(theta_f)
            d_qc_3d[mask] = d_seg

        h_qc_3d = d_qc_3d.copy()

        theta_mesh, r_mesh = np.meshgrid(x_3d, np.linspace(r_inner, r_outer, n_r_3d))
        h_mesh = np.tile(h_qc_3d, (n_r_3d, 1))

        X_mesh = r_mesh * np.cos(theta_mesh)
        Y_mesh = r_mesh * np.sin(theta_mesh)

        ax6.plot_surface(X_mesh, Y_mesh, h_mesh, cmap='coolwarm', alpha=0.85,
                         edgecolor='none', antialiased=True, rstride=1, cstride=1)

        theta_wall = np.linspace(0, 2 * np.pi, n_theta_3d)
        z_wall = np.linspace(0, 1, 2)
        theta_w, z_w = np.meshgrid(theta_wall, z_wall)
        x_outer = r_outer * np.cos(theta_w)
        y_outer = r_outer * np.sin(theta_w)
        h_outer_wall = np.tile(h_qc_3d, (2, 1)) * z_w
        ax6.plot_surface(x_outer, y_outer, h_outer_wall, color='steelblue', alpha=0.3, edgecolor='none')

        ax6.plot_surface(X_mesh, Y_mesh, np.zeros_like(h_mesh),
                         color='lightgray', alpha=0.2, edgecolor='none')

        ax6.set_xlabel("X / cm", fontsize=9)
        ax6.set_ylabel("Y / cm", fontsize=9)
        ax6.set_zlabel("Distance / cm", fontsize=9)
        ax6.set_title("3D Annular Pie — Height Variation (2 cycles)\n(Inner ⌀=5cm, Outer ⌀=10cm)",
                       fontsize=13, fontweight='bold', pad=10)
        ax6.view_init(elev=30, azim=-60)
        fig_combined.tight_layout()
        st.pyplot(fig_combined)
        plt.close(fig_combined)

        # Platform flatness analysis
        platform_bio = bio_spread[(InterDistance > sp) & (InterDistance < ep)]
        pre_platform_bio = bio_spread[InterDistance < sp]
        if len(platform_bio) > 0 and len(pre_platform_bio) > 0:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            flat_cols = st.columns(3)
            with flat_cols[0]:
                st.metric("Platform Std Dev", f"{np.std(platform_bio):.6f}")
            with flat_cols[1]:
                st.metric("Platform Mean Dose", f"{np.mean(platform_bio):.4f}")
            with flat_cols[2]:
                cv = np.std(platform_bio) / np.mean(platform_bio) * 100 if np.mean(platform_bio) != 0 else 0
                st.metric("Platform CV (%)", f"{cv:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Download results
        st.markdown("---")
        dl_cols = st.columns(3)
        with dl_cols[0]:
            result_df = pd.DataFrame({
                'Depth/cm': InterDistance,
                'BiologyDose_SOBP': bio_spread,
                'PhysicalDose_SOBP': phys_spread,
            })
            buf = io.BytesIO()
            result_df.to_excel(buf, index=False, engine='openpyxl')
            st.download_button(
                "📥 Download SOBP Data (Excel)",
                buf.getvalue(),
                file_name="SOBP_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with dl_cols[1]:
            st.download_button(
                "📥 Download Parameters (TXT)",
                f"Optimized para: {para.tolist()}\nStartPoint: {sp}\nEndPoint: {ep}\nStep: {res['step']}",
                file_name="SOBP_parameters.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl_cols[2]:
            qc_df = pd.DataFrame({
                'Time_rad': x,
                'QuarterCircle_Ideal_cm': d_qc,
                'Optimized_Distance_cm': np.interp(x, t_arr * 2 * np.pi, d_actual),
            })
            buf2 = io.BytesIO()
            qc_df.to_excel(buf2, index=False, engine='openpyxl')
            st.download_button(
                "📥 Download Quarter-Circle Distance (Excel)",
                buf2.getvalue(),
                file_name="quarter_circle_distance.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    else:
        st.info("Click **▶ Run Optimization** in the sidebar to start the SOBP analysis.")

# ---- Tab 2: Simulation ----
with tab_sim:
    # Simulation content
    st.markdown("Heavy ion beam (cylinder) passes through the ring and hits the tumor below. As the ring rotates, the beam penetration depth shifts up and down — completing the SOBP process.")

    if st.session_state.results is not None:
        res = st.session_state.results
        para = np.array(res['para'])
        sp = res['start_point']
        ep = res['end_point']

        sim_btn = st.button("▶ Generate Simulation", type="primary", key="sim_btn")

        if sim_btn or st.session_state.get('sim_generated', False):
            if sim_btn:
                st.session_state.sim_generated = True

            with st.spinner("Generating radiation simulation..."):
                sim_progress = st.progress(0, text="Preparing simulation...")

                bragg = Bragg(InterDistance, PristinePP, let, rbe, 1, res['step'], sp, ep)
                t_arr = bragg.Time()
                d_actual = np.cumsum(bragg.Velocity(para, t_arr)) * res['step']
                D_max = float(np.max(d_actual))

                sim_progress.progress(10, text="Computing 3D geometry...")

                # Animation parameters (original high quality)
                n_frames = 45
                beam_z_top = D_max * 2.2
                r_inner = 5.0
                r_outer = 10.0
                sobp_width = float(ep - sp)
                beam_cyl_radius = 0.6
                tumor_radius = beam_cyl_radius
                gap_below_ring = 4.0
                tumor_z_top = -gap_below_ring
                tumor_z_bottom = tumor_z_top - sobp_width
                bragg_peak_shallow = tumor_z_top
                bragg_peak_deep = tumor_z_bottom
                beam_world_angle = 0.0
                beam_cx = 7.5
                beam_cy = 0.0

                # Pre-compute irregular tumor mesh (deformed ellipsoid)
                n_tumor_th = 30
                n_tumor_ph = 20
                tumor_th_s = np.linspace(0, 2 * np.pi, n_tumor_th)
                tumor_ph_s = np.linspace(0, np.pi, n_tumor_ph)
                tumor_TH, tumor_PH = np.meshgrid(tumor_th_s, tumor_ph_s)
                # Low-frequency perturbations for organic lumpy shape (clamped ≤1 so tumor stays inside beam)
                tumor_perturb = (1.0
                    + 0.08 * np.sin(2 * tumor_TH) * np.sin(tumor_PH)
                    + 0.06 * np.cos(3 * tumor_TH) * np.sin(2 * tumor_PH)
                    + 0.05 * np.sin(tumor_TH + 1) * np.cos(tumor_PH)
                    + 0.04 * np.cos(tumor_TH + 2) * np.sin(3 * tumor_PH)
                )
                tumor_perturb = np.clip(tumor_perturb, 0.7, 1.0)
                tumor_rx = beam_cyl_radius * 0.85
                tumor_ry = beam_cyl_radius * 0.85
                tumor_rz = sobp_width / 2 * 0.95
                tumor_x_irr = tumor_rx * tumor_perturb * np.sin(tumor_PH) * np.cos(tumor_TH)
                tumor_y_irr = tumor_ry * tumor_perturb * np.sin(tumor_PH) * np.sin(tumor_TH)
                tumor_z_center = (tumor_z_top + tumor_z_bottom) / 2
                tumor_z_irr = tumor_rz * tumor_perturb * np.cos(tumor_PH) + tumor_z_center
                beam_th = np.linspace(0, 2 * np.pi, 10)
                z_wall_2 = np.linspace(0, 1, 2)
                n_th_anim = 100
                n_r_anim = 10
                x_anim = np.linspace(0, 2 * np.pi, n_th_anim)
                d_qc_anim = np.zeros(n_th_anim)
                for i in range(2):
                    x0 = i * np.pi
                    mask = (x_anim >= x0) & (x_anim < x0 + np.pi)
                    tau = (x_anim[mask] - x0) / np.pi
                    d_seg = np.zeros(int(np.sum(mask)))
                    rise = tau <= 0.5
                    fall = tau > 0.5
                    s_r = tau[rise] * 2
                    theta_r = 3 * np.pi / 2 - s_r * np.pi / 2
                    d_seg[rise] = D_max + D_max * np.sin(theta_r)
                    s_f = (tau[fall] - 0.5) * 2
                    theta_f = np.pi + s_f * np.pi / 2
                    d_seg[fall] = D_max + D_max * np.sin(theta_f)
                    d_qc_anim[mask] = d_seg
                h_qc_anim = d_qc_anim.copy()
                h_max_3d = float(np.max(h_qc_anim))

                sim_progress.progress(20, text="Rendering frames...")

                fig7 = plt.figure(figsize=(10, 9))
                ax7 = fig7.add_subplot(111, projection='3d')

                gif_frames = []
                tmp_dir_anim = tempfile.mkdtemp()

                for frame in range(n_frames):
                    pct = 20 + int(frame / n_frames * 70)
                    sim_progress.progress(pct, text=f"Rendering frame {frame+1}/{n_frames}...")

                    ax7.cla()
                    rot_angle = 2 * np.pi * frame / n_frames

                    x_rot = x_anim + rot_angle
                    theta_m, r_m = np.meshgrid(x_rot, np.linspace(r_inner, r_outer, n_r_anim))
                    h_m = np.tile(h_qc_anim, (n_r_anim, 1))
                    X_r = r_m * np.cos(theta_m)
                    Y_r = r_m * np.sin(theta_m)

                    ax7.plot_surface(X_r, Y_r, h_m, cmap='coolwarm', alpha=0.75,
                                     edgecolor='none', antialiased=True, rstride=1, cstride=1)

                    theta_wall = x_anim + rot_angle
                    theta_w2, z_w2 = np.meshgrid(theta_wall, z_wall_2)
                    xw = r_outer * np.cos(theta_w2)
                    yw = r_outer * np.sin(theta_w2)
                    hw = np.tile(h_qc_anim, (2, 1)) * z_w2
                    ax7.plot_surface(xw, yw, hw, color='steelblue', alpha=0.25, edgecolor='none', antialiased=True)

                    beam_angle_local = (beam_world_angle - rot_angle) % (2 * np.pi)
                    idx_beam = int(beam_angle_local / (2 * np.pi) * len(h_qc_anim)) % len(h_qc_anim)
                    h_at_beam = h_qc_anim[idx_beam]
                    if h_max_3d > 0:
                        bragg_frac = 1.0 - (h_at_beam / h_max_3d)
                    else:
                        bragg_frac = 0.5
                    bragg_depth = bragg_peak_shallow + bragg_frac * (bragg_peak_deep - bragg_peak_shallow)

                    # ---- Beam as dense dashed lines ----
                    n_beam_lines = 10
                    beam_angles = np.linspace(0, 2 * np.pi, n_beam_lines, endpoint=False)
                    beam_line_x = beam_cx + beam_cyl_radius * np.cos(beam_angles)
                    beam_line_y = beam_cy + beam_cyl_radius * np.sin(beam_angles)
                    # Central line
                    beam_line_x = np.append(beam_line_x, beam_cx)
                    beam_line_y = np.append(beam_line_y, beam_cy)
                    dash_style = (0, (3, 2))  # dense dashed: 3pt dash, 2pt gap

                    # Segment 1: above ring (h_at_beam → beam_z_top)
                    for li in range(len(beam_line_x)):
                        ax7.plot([beam_line_x[li], beam_line_x[li]],
                                 [beam_line_y[li], beam_line_y[li]],
                                 [h_at_beam, beam_z_top],
                                 color='black', linewidth=1.0, linestyle=dash_style, alpha=0.35)

                    # Segment 2: in ring (0 → h_at_beam)
                    for li in range(len(beam_line_x)):
                        ax7.plot([beam_line_x[li], beam_line_x[li]],
                                 [beam_line_y[li], beam_line_y[li]],
                                 [0, h_at_beam],
                                 color='black', linewidth=1.0, linestyle=dash_style, alpha=0.6)

                    # Segment 3: gap (tumor_z_top → 0)
                    for li in range(len(beam_line_x)):
                        ax7.plot([beam_line_x[li], beam_line_x[li]],
                                 [beam_line_y[li], beam_line_y[li]],
                                 [tumor_z_top, 0],
                                 color='black', linewidth=1.0, linestyle=dash_style, alpha=0.3)

                    # Segment 4: in tumor (tumor_z_top → bragg_depth)
                    for li in range(len(beam_line_x)):
                        ax7.plot([beam_line_x[li], beam_line_x[li]],
                                 [beam_line_y[li], beam_line_y[li]],
                                 [tumor_z_top, bragg_depth],
                                 color='black', linewidth=1.5, linestyle=dash_style, alpha=0.9)

                    # Beam shadow on ring top surface (dark)
                    shadow_r = np.linspace(0, beam_cyl_radius, 3)
                    shadow_th = np.linspace(0, 2 * np.pi, 12)
                    shadow_R, shadow_TH = np.meshgrid(shadow_r, shadow_th)
                    shadow_x = beam_cx + shadow_R * np.cos(shadow_TH)
                    shadow_y = beam_cy + shadow_R * np.sin(shadow_TH)
                    shadow_z = np.full_like(shadow_x, h_at_beam)
                    ax7.plot_surface(shadow_x, shadow_y, shadow_z, color='black', alpha=0.5, edgecolor='none', antialiased=True)

                    ax7.text(beam_cx + beam_cyl_radius + 1.0, beam_cy, beam_z_top * 0.9,
                             "ION\nBEAM", color='black', fontsize=8, fontweight='bold')

                    # Irregular tumor (flesh-colored)
                    ax7.plot_surface(beam_cx + tumor_x_irr, beam_cy + tumor_y_irr, tumor_z_irr,
                                     color='#E8A0A0', alpha=0.55, edgecolor='none', antialiased=True)

                    # Bragg peak curve
                    bp_x_offset = beam_cx + tumor_radius + 2.0
                    n_bp_pts = 50
                    bp_depth_arr = np.linspace(sp, ep, n_bp_pts)
                    bp_dose_arr = PP(bp_depth_arr)
                    bp_dose_min = float(np.min(bp_dose_arr))
                    bp_dose_max = float(np.max(bp_dose_arr))
                    if bp_dose_max > bp_dose_min:
                        bp_dose_norm = (bp_dose_arr - bp_dose_min) / (bp_dose_max - bp_dose_min)
                    else:
                        bp_dose_norm = np.zeros(n_bp_pts)
                    bp_z_arr = tumor_z_top + (bp_depth_arr - sp) / (ep - sp) * (tumor_z_bottom - tumor_z_top)
                    peak_idx = int(np.argmax(bp_dose_norm))
                    peak_z_original = bp_z_arr[peak_idx]
                    z_shift = bragg_depth - peak_z_original
                    bp_z_shifted = np.clip(bp_z_arr + z_shift, tumor_z_bottom, tumor_z_top)
                    bp_curve_width = 2.5
                    bp_x_arr = bp_x_offset + bp_dose_norm * bp_curve_width
                    ax7.plot(bp_x_arr, np.full(n_bp_pts, beam_cy), bp_z_shifted,
                             color='#FF6600', linewidth=2.0, alpha=0.9, zorder=20)
                    ax7.plot([bp_x_offset, bp_x_offset], [beam_cy, beam_cy],
                             [tumor_z_top, tumor_z_bottom],
                             color='gray', linewidth=0.8, alpha=0.5, linestyle='--')
                    ax7.plot([bp_x_offset, bp_x_offset + bp_curve_width], [beam_cy, beam_cy],
                             [bragg_depth, bragg_depth],
                             color='#FF6600', linewidth=1.5, alpha=0.8, linestyle='-')
                    ax7.text(bp_x_offset + bp_curve_width + 0.5, beam_cy, bragg_depth,
                             "Bragg\npeak", color='#FF6600', fontsize=7, fontweight='bold')

                    # ---- Mini SOBP curve (with rising edge) to the right of "Bragg peak" label ----
                    sobp_x_offset = bp_x_offset + bp_curve_width + 3.5
                    sobp_curve_width = 1.8  # smaller than pristine
                    # Get SOBP data (biological dose) — include rising edge + plateau
                    n_sobp_pts = 120
                    bio_data = np.array(st.session_state.results['bio_spread'])
                    # Range: from 0 to ep, to include rising edge before sp
                    sobp_depth_sampled = np.linspace(0, ep, n_sobp_pts)
                    bio_interp = np.interp(sobp_depth_sampled, InterDistance, bio_data)
                    if len(bio_interp) > 2:
                        # Normalize using plateau region only (so plateau fills full width)
                        plateau_mask = (sobp_depth_sampled >= sp) & (sobp_depth_sampled <= ep)
                        plateau_vals = bio_interp[plateau_mask]
                        if len(plateau_vals) > 0:
                            sobp_v_min = 0.0  # rising edge starts from 0
                            sobp_v_max = float(np.max(plateau_vals))
                        else:
                            sobp_v_min = 0.0
                            sobp_v_max = float(np.max(bio_interp))
                        if sobp_v_max > 0:
                            sobp_v_norm = bio_interp / sobp_v_max
                            sobp_v_norm = np.clip(sobp_v_norm, 0, 1)
                        else:
                            sobp_v_norm = np.zeros(n_sobp_pts)
                        # Map depth to z: 0→above tumor top (rising edge), sp→tumor_z_top, ep→tumor_z_bottom
                        # Rising edge: depth 0→sp maps to z above tumor_z_top
                        rising_height = 1.5  # cm above tumor top for rising edge display
                        sobp_z_mapped = np.where(
                            sobp_depth_sampled <= sp,
                            tumor_z_top + rising_height * (1.0 - sobp_depth_sampled / sp),
                            tumor_z_top + (sobp_depth_sampled - sp) / (ep - sp) * (tumor_z_bottom - tumor_z_top)
                        )
                        sobp_x_arr = sobp_x_offset + sobp_v_norm * sobp_curve_width
                        ax7.plot(sobp_x_arr, np.full(n_sobp_pts, beam_cy), sobp_z_mapped,
                                 color='#0066CC', linewidth=1.0, alpha=0.85, zorder=20)
                        # Vertical reference line: sp→tumor_z_top, ep→tumor_z_bottom
                        ax7.plot([sobp_x_offset, sobp_x_offset], [beam_cy, beam_cy],
                                 [tumor_z_top, tumor_z_bottom],
                                 color='gray', linewidth=0.6, alpha=0.4, linestyle='--')
                        # SOBP label
                        ax7.text(sobp_x_offset + sobp_curve_width + 0.3, beam_cy,
                                 (tumor_z_top + tumor_z_bottom) / 2,
                                 "SOBP", color='#0066CC', fontsize=7, fontweight='bold')

                    ax7.text(beam_cx - tumor_radius - 4.5, beam_cy, tumor_z_bottom * 0.7,
                             f"TUMOR\n({sobp_width:.2f}cm)", color='#8B3A3A', fontsize=8, fontweight='bold')

                    ax7.set_xlim(-r_outer * 1.3, r_outer * 1.3)
                    ax7.set_ylim(-r_outer * 1.3, r_outer * 1.3)
                    ax7.set_zlim(tumor_z_bottom * 1.3, beam_z_top * 1.1)
                    ax7.set_xlabel("X / cm", fontsize=8)
                    ax7.set_ylabel("Y / cm", fontsize=8)
                    ax7.set_zlabel("Depth / cm", fontsize=8)
                    ax7.set_title(f"SOBP Radiation Simulation (Frame {frame+1}/{n_frames})\n"
                                   f"Ring height: {h_at_beam:.2f} cm  |  Bragg peak depth: {bragg_depth:.2f} cm  |  Tumor: {sobp_width:.2f} cm",
                                   fontsize=9, fontweight='bold')
                    ax7.view_init(elev=25, azim=-120)

                    fpath = os.path.join(tmp_dir_anim, f'frame_{frame:03d}.png')
                    fig7.savefig(fpath, dpi=120, bbox_inches='tight')
                    gif_frames.append(fpath)

                plt.close(fig7)

                sim_progress.progress(92, text="Building GIF...")

                from PIL import Image
                gif_path = os.path.join(tmp_dir_anim, 'sobp_simulation.gif')
                imgs = [Image.open(f) for f in gif_frames]
                imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=120, loop=0)
                for img in imgs:
                    img.close()

                sim_progress.progress(100, text="Done!")
                time.sleep(0.3)
                sim_progress.empty()

                with open(gif_path, 'rb') as f:
                    gif_bytes = f.read()
                st.image(gif_bytes, caption="Heavy Ion Beam + Rotating Ring → SOBP", use_container_width=True)

                st.download_button(
                    "📥 Download Animation (GIF)",
                    gif_bytes,
                    file_name="SOBP_simulation.gif",
                    mime="image/gif",
                )

                # Convert GIF to MP4 video using ffmpeg
                mp4_path = os.path.join(tmp_dir_anim, 'sobp_simulation.mp4')
                try:
                    import subprocess
                    fps = max(1, round(1000.0 / 120))  # duration=120ms per frame → ~8fps
                    vf_filter = f'fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2'
                    result = subprocess.run(
                        ['ffmpeg', '-y', '-i', gif_path, '-vf', vf_filter, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4_path],
                        capture_output=True, timeout=60
                    )
                    if result.returncode == 0 and os.path.exists(mp4_path):
                        with open(mp4_path, 'rb') as vf:
                            mp4_bytes = vf.read()
                        st.download_button(
                            "🎬 Download Animation (MP4 Video)",
                            mp4_bytes,
                            file_name="SOBP_simulation.mp4",
                            mime="video/mp4",
                        )
                    else:
                        st.caption(f"(MP4 conversion failed: rc={result.returncode}, err={result.stderr.decode()[:200]})")
                except Exception as e:
                    st.caption(f"(MP4 conversion error: {e})")

                import shutil
                shutil.rmtree(tmp_dir_anim, ignore_errors=True)
    else:
        st.warning("Please run optimization first (sidebar → ▶ Run Optimization) before generating simulation.")

# ---- Tab 3: Help ----
with tab_help:
    # Help content

    with st.expander("1. What is SOBP?", expanded=True):
        st.markdown("""
        **Spread Out Bragg Peak (SOBP)** is a technique used in heavy ion radiotherapy to create a uniform dose distribution across a tumor volume.

        A single pristine Bragg peak deposits maximum dose at a specific depth. By modulating the beam energy (i.e., varying the particle velocity over time), multiple Bragg peaks at different depths are superimposed to form a flat "plateau" — the SOBP — that covers the entire tumor thickness.
        """)

    with st.expander("2. How to use this app"):
        st.markdown("""
        **Step 1 — Load Data:** In the sidebar, choose "Demo Data" (built-in example) or "Upload Files" to provide your own BraggPeak and RBE data files.

        **Step 2 — Set Interpolation Step:** Adjust the step size (default 0.001 cm). Smaller values give finer resolution but are slower.

        **Step 3 — Define Spread Width:** Set StartPoint and EndPoint (in cm) to define the SOBP plateau region.

        **Step 4 — Run Optimization:** Click "▶ Run Optimization". The app uses `scipy.optimize.fmin` to minimize the standard deviation of the dose within the platform region.

        **Step 5 — View Results:** Switch to the "Optimization Results" tab to see:
        - Optimized velocity parameters
        - SOBP dose curves (biological & physical)
        - Velocity & Distance vs Time
        - Quarter-circle distance comparison
        - 3D annular pie visualization
        - Platform flatness metrics
        - Download options

        **Step 6 — Simulation:** Switch to the "Simulation" tab and click "▶ Generate Simulation" to create a 3D animated GIF of the radiation process.
        """)

    with st.expander("3. Data Format"):
        st.markdown("""
        **BraggPeak file** (xlsx/csv): Two columns — Depth (cm) and Pristine peak value.

        | Depth/cm | Pristine |
        |----------|----------|
        | 0.00     | 0.001    |
        | 0.01     | 0.002    |
        | ...      | ...      |

        **RBE file** (xlsx/csv): Two columns — LET (Kev/μm) and RBE value.

        | LET/Kev | RBE |
        |---------|-----|
        | 10      | 1.2 |
        | 20      | 1.5 |
        | ...     | ... |
        """)

    with st.expander("4. Algorithm Details"):
        st.markdown("""
        The optimization uses a 4th-order polynomial velocity function:

        `v(t) = a₀t⁴ + a₁t³ + a₂t² + a₃t + a₄`

        The cumulative distance determines which portion of the Bragg peak contributes at each time step. The objective function minimizes the standard deviation of the biological dose within the platform region [StartPoint, EndPoint].

        The quarter-circle distance model provides an idealized reference: the distance rises and falls following quarter-circle arcs, creating smooth, symmetric peaks — 2 cycles over [0, 2π].
        """)

    with st.expander("5. 3D Visualization"):
        st.markdown("""
        **Annular Pie:** A 3D ring (inner ⌀=5cm, outer ⌀=10cm) whose height varies with angle, representing the quarter-circle distance profile. The height determines how much the beam is attenuated before reaching the tumor.

        **Simulation:** The ring rotates while a fixed ion beam passes through it. The beam penetration depth (Bragg peak) moves up and down within the tumor as the ring's thickness at the beam position changes. The orange Bragg peak curve on the right moves in sync with the beam's depth.
        """)

    with st.expander("6. References"):
        st.markdown("""
        - ZHOU Qing-Ming, ZHANG Hong, LI Qiang, WEI Li-Li, LIU Xin-Guo, GU Meng-Xia, WU Gui-Hong, DAI Zhong-Yin. Rotating wheel filter design and optimization for a uniform depth dose distribution in heavy ion therapy[J]. *Chinese Physics C*, 2008, 32(S2): 294-298.
        """)

    with st.expander("7. Contact"):
        st.markdown("""
        WeChat: **icecoler**

        For questions, suggestions, or bug reports, feel free to reach out.
        """)
