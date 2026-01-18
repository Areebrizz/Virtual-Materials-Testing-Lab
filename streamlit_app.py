"""
STREAMLIT INTERFACE FOR VIRTUAL MATERIALS LAB
Advanced Web-Based Materials Science Simulator
"""

import streamlit as st
import sys
import os

# Add the app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import VirtualMaterialsLab, CrystalStructure, MaterialProperties, Microstructure, HeatTreatment
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Virtual Materials Testing Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .material-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #3498db;
    }
    .property-value {
        font-size: 1.1rem;
        font-weight: 500;
        color: #2c3e50;
    }
    .download-button {
        background-color: #3498db;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'lab' not in st.session_state:
    st.session_state.lab = VirtualMaterialsLab()
if 'current_material' not in st.session_state:
    st.session_state.current_material = None
if 'current_microstructure' not in st.session_state:
    st.session_state.current_microstructure = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}

# Main header
st.markdown('<h1 class="main-header">🔬 Virtual Materials Testing Laboratory</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-bottom: 2rem;'>
    <i>Version 3.0 | ISO 6892-1 Compliant | Multi-scale Materials Science Simulator</i>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.markdown("## 🧪 Navigation")
    page = st.radio(
        "Select Module:",
        ["🏠 Dashboard", 
         "⚗️ Sample Preparation", 
         "📈 Tensile Testing",
         "🔄 Fatigue Testing",
         "⚡ Fracture Toughness",
         "🔥 Creep Testing",
         "🔬 Microstructure Viewer",
         "🧪 Alloy Designer",
         "📊 Data Export"]
    )
    
    st.markdown("---")
    st.markdown("### 📋 Quick Stats")
    
    if st.session_state.current_material:
        material_name = st.session_state.current_material
        st.metric("Current Material", material_name)
    
    if st.session_state.current_microstructure:
        grain_size = st.session_state.current_microstructure.grain_size
        st.metric("Grain Size", f"{grain_size:.1f} μm")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    # Theme selector
    theme = st.selectbox(
        "Plot Theme",
        ["plotly", "plotly_white", "plotly_dark", "seaborn", "simple_white"]
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.8rem; color: #95a5a6;'>
    <b>Academic Edition</b><br>
    Multi-scale modeling framework<br>
    ISO standards compliant<br>
    Research-grade simulations
    </div>
    """, unsafe_allow_html=True)

# Main content area
if page == "🏠 Dashboard":
    st.markdown('<h2 class="section-header">Laboratory Dashboard</h2>', unsafe_allow_html=True)
    
    # Create columns for dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Available Materials", len(st.session_state.lab.materials_db))
    
    with col2:
        st.metric("Test Modules", "7")
    
    with col3:
        st.metric("ISO Standards", "ISO 6892-1")
    
    # Quick start guide
    st.markdown('<h3 class="subsection-header">Quick Start Guide</h3>', unsafe_allow_html=True)
    
    steps = [
        "1. **Sample Preparation**: Select a material and design its microstructure",
        "2. **Apply Heat Treatment**: Simulate quenching, tempering, and aging processes",
        "3. **Run Tests**: Use any testing module (Tensile, Fatigue, Fracture, Creep)",
        "4. **Analyze Results**: View interactive visualizations and export data"
    ]
    
    for step in steps:
        st.markdown(step)
    
    # Material database preview
    st.markdown('<h3 class="subsection-header">Material Database</h3>', unsafe_allow_html=True)
    
    materials_data = []
    for name, props in st.session_state.lab.materials_db.items():
        materials_data.append({
            "Material": name,
            "E (GPa)": props.youngs_modulus,
            "σ_y (MPa)": props.yield_strength,
            "σ_UTS (MPa)": props.tensile_strength,
            "ε_f (%)": props.elongation,
            "K_IC (MPa√m)": props.fracture_toughness
        })
    
    df_materials = pd.DataFrame(materials_data)
    st.dataframe(df_materials, use_container_width=True, hide_index=True)

elif page == "⚗️ Sample Preparation":
    st.markdown('<h2 class="section-header">Sample Preparation Station</h2>', unsafe_allow_html=True)
    
    # Material selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="subsection-header">Material Selection</h3>', unsafe_allow_html=True)
        material_options = list(st.session_state.lab.materials_db.keys())
        selected_material = st.selectbox(
            "Choose Base Material",
            material_options,
            index=0,
            help="Select from predefined materials database"
        )
        
        if st.button("Load Material", type="primary"):
            st.session_state.current_material = selected_material
            st.success(f"Loaded {selected_material}")
            st.rerun()
    
    with col2:
        if st.session_state.current_material:
            material = st.session_state.lab.materials_db[st.session_state.current_material]
            st.markdown('<h3 class="subsection-header">Material Properties</h3>', unsafe_allow_html=True)
            
            # Display material properties in a nice format
            props = [
                ("Young's Modulus", f"{material.youngs_modulus} GPa"),
                ("Yield Strength", f"{material.yield_strength} MPa"),
                ("Tensile Strength", f"{material.tensile_strength} MPa"),
                ("Elongation", f"{material.elongation} %"),
                ("Fracture Toughness", f"{material.fracture_toughness} MPa√m"),
                ("Crystal Structure", material.crystal_structure.value)
            ]
            
            for prop_name, prop_value in props:
                st.markdown(f"**{prop_name}:** {prop_value}")
    
    # Microstructure Designer
    st.markdown('<h3 class="subsection-header">Microstructure Designer</h3>', unsafe_allow_html=True)
    
    if st.session_state.current_material:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            grain_size = st.slider(
                "Grain Size (μm)",
                min_value=1.0,
                max_value=500.0,
                value=50.0,
                step=1.0,
                help="Average grain diameter in micrometers"
            )
            
            porosity = st.slider(
                "Porosity (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="Volume fraction of pores"
            )
        
        with col2:
            inclusion_size = st.slider(
                "Inclusion Size (μm)",
                min_value=0.1,
                max_value=100.0,
                value=10.0,
                step=0.1
            )
            
            defect_density = st.slider(
                "Defect Density (per mm²)",
                min_value=0.0,
                max_value=10000.0,
                value=1000.0,
                step=100.0
            )
        
        with col3:
            # Phase fractions based on material
            if st.session_state.current_material == "Ti-6Al-4V":
                alpha_fraction = st.slider("Alpha Phase Fraction", 0.0, 1.0, 0.9, 0.01)
                phase_fraction = {"alpha": alpha_fraction, "beta": 1 - alpha_fraction}
            elif "steel" in st.session_state.current_material.lower():
                ferrite_fraction = st.slider("Ferrite Fraction", 0.0, 1.0, 0.85, 0.01)
                phase_fraction = {"ferrite": ferrite_fraction, "pearlite": 1 - ferrite_fraction}
            else:
                phase_fraction = {"matrix": 1.0}
        
        if st.button("Design Microstructure", type="primary"):
            microstructure = st.session_state.lab.design_microstructure(
                st.session_state.current_material,
                grain_size=grain_size,
                phase_fraction=phase_fraction,
                porosity=porosity/100,
                inclusion_size=inclusion_size
            )
            st.session_state.current_microstructure = microstructure
            st.success("Microstructure designed successfully!")
            
            # Display microstructure properties
            with st.expander("View Microstructure Properties"):
                st.write(f"**Grain Size:** {microstructure.grain_size} μm")
                st.write(f"**Hall-Petch Strengthening:** {microstructure.calculate_hall_peetch():.1f} MPa")
                st.write(f"**Phase Fractions:** {microstructure.phase_fractions}")
                st.write(f"**Crystal Structure:** {microstructure.crystal_structure.value}")
    
    # Heat Treatment Simulator
    st.markdown('<h3 class="subsection-header">Heat Treatment Simulator</h3>', unsafe_allow_html=True)
    
    if st.session_state.current_microstructure:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quenching_rate = st.slider(
                "Quenching Rate (°C/s)",
                min_value=1.0,
                max_value=1000.0,
                value=100.0,
                step=10.0
            )
        
        with col2:
            tempering_temp = st.slider(
                "Tempering Temperature (°C)",
                min_value=100.0,
                max_value=700.0,
                value=600.0,
                step=10.0
            )
        
        with col3:
            tempering_time = st.slider(
                "Tempering Time (hours)",
                min_value=0.5,
                max_value=24.0,
                value=2.0,
                step=0.5
            )
        
        if st.button("Apply Heat Treatment", type="primary"):
            heat_treatment = st.session_state.lab.apply_heat_treatment(
                quenching_rate=quenching_rate,
                tempering_temp=tempering_temp,
                tempering_time=tempering_time
            )
            st.session_state.current_heat_treatment = heat_treatment
            st.success("Heat treatment applied successfully!")
            
            # Display heat treatment results
            with st.expander("View Heat Treatment Results"):
                st.write(f"**Cooling Medium:** {heat_treatment.cooling_medium}")
                st.write(f"**Hardenability:** {heat_treatment.calculate_hardenability():.1f}")

elif page == "📈 Tensile Testing":
    st.markdown('<h2 class="section-header">Tensile Testing Module</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_material:
        st.warning("⚠️ Please select a material in the Sample Preparation module first!")
        st.stop()
    
    # Testing parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        constitutive_model = st.selectbox(
            "Constitutive Model",
            ["hollomon", "voce"],
            format_func=lambda x: "Hollomon (σ=Kεⁿ)" if x == "hollomon" else "Voce (σ=σ₀+Q(1-exp(-bε)))"
        )
    
    with col2:
        temperature = st.slider(
            "Test Temperature (°C)",
            min_value=20.0,
            max_value=800.0,
            value=20.0,
            step=10.0
        )
    
    with col3:
        strain_rate = st.select_slider(
            "Strain Rate (s⁻¹)",
            options=[0.0001, 0.001, 0.01, 0.1, 1.0],
            value=0.001
        )
    
    # Initialize tester
    tensile_tester = st.session_state.lab.TensileTester(
        st.session_state.lab.materials_db[st.session_state.current_material],
        st.session_state.current_microstructure
    )
    
    # Run test
    if st.button("Run Tensile Test", type="primary"):
        with st.spinner("Running tensile test simulation..."):
            eps, stress = tensile_tester.generate_stress_strain_curve(
                constitutive_model=constitutive_model,
                temperature=temperature,
                strain_rate=strain_rate
            )
            
            # Calculate properties
            properties = tensile_tester.calculate_mechanical_properties()
            st.session_state.test_results['tensile'] = properties
            
            # Display results in columns
            st.markdown('<h3 class="subsection-header">Mechanical Properties</h3>', unsafe_allow_html=True)
            
            cols = st.columns(4)
            prop_items = list(properties.items())
            
            for idx, (prop_name, prop_value) in enumerate(prop_items):
                with cols[idx % 4]:
                    st.metric(prop_name, f"{prop_value:.2f}")
            
            # Visualization
            st.markdown('<h3 class="subsection-header">Stress-Strain Curves</h3>', unsafe_allow_html=True)
            
            fig = tensile_tester.visualize_curve(show_true=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data download
            st.markdown('<h3 class="subsection-header">Export Data</h3>', unsafe_allow_html=True)
            
            # Create dataframe for download
            df_tensile = pd.DataFrame({
                'Strain (%)': eps * 100,
                'Engineering Stress (MPa)': stress,
                'True Strain (%)': np.log(1 + eps) * 100,
                'True Stress (MPa)': stress * (1 + eps)
            })
            
            # Convert to CSV
            csv = df_tensile.to_csv(index=False)
            st.download_button(
                label="Download CSV Data",
                data=csv,
                file_name="tensile_test_data.csv",
                mime="text/csv"
            )
    
    # Show previous results if available
    if 'tensile' in st.session_state.test_results:
        st.markdown("---")
        st.markdown('<h3 class="subsection-header">Previous Test Results</h3>', unsafe_allow_html=True)
        
        df_prev = pd.DataFrame([st.session_state.test_results['tensile']])
        st.dataframe(df_prev, use_container_width=True, hide_index=True)

elif page == "🔄 Fatigue Testing":
    st.markdown('<h2 class="section-header">Fatigue Testing Module</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_material:
        st.warning("⚠️ Please select a material in the Sample Preparation module first!")
        st.stop()
    
    # Tabs for different fatigue analyses
    tab1, tab2, tab3 = st.tabs(["S-N Curve", "Crack Growth", "Fracture Surface"])
    
    with tab1:
        st.markdown('<h3 class="subsection-header">S-N Curve Generator</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            R_ratio = st.slider(
                "R-ratio (σ_min/σ_max)",
                min_value=-1.0,
                max_value=0.5,
                value=-1.0,
                step=0.1
            )
        
        with col2:
            surface_finish = st.selectbox(
                "Surface Finish",
                ["polished", "machined", "hot_rolled", "as_forged"]
            )
        
        with col3:
            reliability = st.slider(
                "Reliability Level",
                min_value=0.50,
                max_value=0.99,
                value=0.95,
                step=0.01
            )
        
        if st.button("Generate S-N Curve", type="primary"):
            fatigue_tester = st.session_state.lab.FatigueTester(
                st.session_state.lab.materials_db[st.session_state.current_material],
                st.session_state.current_microstructure
            )
            
            with st.spinner("Generating S-N curve..."):
                N_cycles, stress_amp = fatigue_tester.generate_SN_curve(
                    R_ratio=R_ratio,
                    surface_finish=surface_finish,
                    reliability=reliability
                )
                
                # Create S-N curve plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=np.log10(N_cycles),
                    y=stress_amp,
                    mode='lines+markers',
                    name='S-N Curve',
                    line=dict(color='blue', width=2)
                ))
                
                # Add fatigue limit line
                fatigue_limit = st.session_state.lab.materials_db[st.session_state.current_material].fatigue_limit
                fig.add_hline(
                    y=fatigue_limit,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Fatigue Limit: {fatigue_limit:.1f} MPa"
                )
                
                fig.update_layout(
                    title="S-N Curve (Stress-Life)",
                    xaxis_title="Log Cycles to Failure (N)",
                    yaxis_title="Stress Amplitude (MPa)",
                    height=500,
                    template=theme
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data download
                df_sn = pd.DataFrame({
                    'Cycles': N_cycles,
                    'Stress_Amplitude_MPa': stress_amp
                })
                
                csv = df_sn.to_csv(index=False)
                st.download_button(
                    label="Download S-N Data",
                    data=csv,
                    file_name="sn_curve_data.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.markdown('<h3 class="subsection-header">Paris Law Crack Growth</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_crack = st.number_input(
                "Initial Crack Size (mm)",
                min_value=0.01,
                max_value=5.0,
                value=0.1,
                step=0.01
            )
            
            final_crack = st.number_input(
                "Final Crack Size (mm)",
                min_value=0.1,
                max_value=50.0,
                value=10.0,
                step=0.1
            )
        
        with col2:
            delta_K_th = st.number_input(
                "ΔK threshold (MPa√m)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.1
            )
        
        if st.button("Simulate Crack Growth", type="primary"):
            fatigue_tester = st.session_state.lab.FatigueTester(
                st.session_state.lab.materials_db[st.session_state.current_material],
                st.session_state.current_microstructure
            )
            
            with st.spinner("Running crack growth simulation..."):
                a, da_dN, N = fatigue_tester.paris_law_crack_growth(
                    initial_crack=initial_crack,
                    final_crack=final_crack,
                    delta_K_th=delta_K_th
                )
                
                # Create crack growth plot
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Crack Growth Rate", "Crack Length vs Cycles"),
                    horizontal_spacing=0.2
                )
                
                # da/dN vs ΔK plot
                delta_sigma = 200  # Constant amplitude loading
                delta_K = delta_sigma * np.sqrt(np.pi * a)
                
                fig.add_trace(
                    go.Scatter(x=delta_K, y=da_dN, mode='lines',
                              name='da/dN', line=dict(color='red', width=2)),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=[delta_K_th, delta_K_th], y=[1e-8, 1e-3],
                              mode='lines', name='Threshold',
                              line=dict(dash='dash', color='green')),
                    row=1, col=1
                )
                
                # Crack length vs cycles
                fig.add_trace(
                    go.Scatter(x=N, y=a, mode='lines',
                              name='Crack Growth', line=dict(color='blue', width=2)),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="ΔK (MPa√m)", type="log", row=1, col=1)
                fig.update_yaxes(title_text="da/dN (mm/cycle)", type="log", row=1, col=1)
                fig.update_xaxes(title_text="Cycles (N)", row=1, col=2)
                fig.update_yaxes(title_text="Crack Length (mm)", row=1, col=2)
                
                fig.update_layout(height=500, template=theme, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display final results
                if len(N) > 0:
                    st.info(f"**Total cycles to failure:** {N[-1]:,.0f} cycles")
    
    with tab3:
        st.markdown('<h3 class="subsection-header">Fracture Surface Simulation</h3>', unsafe_allow_html=True)
        
        crack_length = st.slider(
            "Crack Length (mm)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        )
        
        if st.button("Generate Fracture Surface", type="primary"):
            fatigue_tester = st.session_state.lab.FatigueTester(
                st.session_state.lab.materials_db[st.session_state.current_material],
                st.session_state.current_microstructure
            )
            
            with st.spinner("Generating 3D fracture surface..."):
                fig = fatigue_tester.fracture_surface_simulation(crack_length=crack_length)
                st.plotly_chart(fig, use_container_width=True)

elif page == "⚡ Fracture Toughness":
    st.markdown('<h2 class="section-header">Fracture Toughness Testing</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_material:
        st.warning("⚠️ Please select a material in the Sample Preparation module first!")
        st.stop()
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        K_I = st.slider(
            "Stress Intensity Factor K_I (MPa√m)",
            min_value=10.0,
            max_value=100.0,
            value=40.0,
            step=5.0
        )
    
    with col2:
        plane_condition = st.radio(
            "Stress Condition",
            ["Plane Stress", "Plane Strain"],
            horizontal=True
        )
    
    if st.button("Analyze Crack Tip", type="primary"):
        fracture_tester = st.session_state.lab.FractureToughnessTester(
            st.session_state.lab.materials_db[st.session_state.current_material]
        )
        
        with st.spinner("Calculating stress fields..."):
            # Calculate plastic zone
            plane_stress = True if plane_condition == "Plane Stress" else False
            r_p = fracture_tester.estimate_plastic_zone(K_I=K_I, plane_stress=plane_stress)
            
            # Display results
            st.metric("Plastic Zone Size", f"{r_p:.3f} mm")
            
            # Show stress field visualization
            fig = fracture_tester.visualize_crack_tip(K_I=K_I)
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔥 Creep Testing":
    st.markdown('<h2 class="section-header">Creep Testing Module</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_material:
        st.warning("⚠️ Please select a material in the Sample Preparation module first!")
        st.stop()
    
    tab1, tab2 = st.tabs(["Creep Deformation", "Stress Rupture"])
    
    with tab1:
        st.markdown('<h3 class="subsection-header">Creep Deformation Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stress = st.number_input(
                "Applied Stress (MPa)",
                min_value=10.0,
                max_value=500.0,
                value=150.0,
                step=10.0
            )
        
        with col2:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=200.0,
                max_value=1200.0,
                value=600.0,
                step=10.0
            )
        
        with col3:
            time_hours = st.number_input(
                "Time (hours)",
                min_value=1.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0
            )
        
        if st.button("Calculate Creep Strain", type="primary"):
            creep_tester = st.session_state.lab.CreepTester(
                st.session_state.lab.materials_db[st.session_state.current_material]
            )
            
            with st.spinner("Calculating creep deformation..."):
                t, eps_creep = creep_tester.creep_deformation(
                    stress=stress,
                    temperature=temperature,
                    time_hours=time_hours
                )
                
                # Create creep curve
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=t,
                    y=eps_creep,
                    mode='lines',
                    name='Creep Strain',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"Creep Curve at {stress} MPa, {temperature}°C",
                    xaxis_title="Time (hours)",
                    yaxis_title="Creep Strain (%)",
                    height=500,
                    template=theme
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display final strain
                st.info(f"**Final creep strain:** {eps_creep[-1]:.3f}%")
    
    with tab2:
        st.markdown('<h3 class="subsection-header">Stress Rupture Curves</h3>', unsafe_allow_html=True)
        
        temperature_rupture = st.slider(
            "Temperature for Rupture Analysis (°C)",
            min_value=400.0,
            max_value=1000.0,
            value=600.0,
            step=50.0
        )
        
        if st.button("Generate Rupture Curves", type="primary"):
            creep_tester = st.session_state.lab.CreepTester(
                st.session_state.lab.materials_db[st.session_state.current_material]
            )
            
            with st.spinner("Generating stress rupture curves..."):
                fig = creep_tester.stress_rupture_curve(temperature=temperature_rupture)
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔬 Microstructure Viewer":
    st.markdown('<h2 class="section-header">Microstructure Viewer</h2>', unsafe_allow_html=True)
    
    # Microstructure parameters
    col1, col2 = st.columns(2)
    
    with col1:
        grain_size = st.slider(
            "Grain Size for Visualization (μm)",
            min_value=1.0,
            max_value=200.0,
            value=30.0,
            step=1.0
        )
    
    with col2:
        if st.session_state.current_material == "Ti-6Al-4V":
            alpha_fraction = st.slider("Alpha Phase Fraction", 0.0, 1.0, 0.9, 0.01)
            phase_fractions = {'alpha': alpha_fraction, 'beta': 1 - alpha_fraction}
        else:
            phase_fractions = {'matrix': 1.0}
    
    tab1, tab2 = st.tabs(["3D Visualization", "EBSD Simulation"])
    
    with tab1:
        if st.button("Generate 3D Microstructure", type="primary"):
            microstructure_viewer = st.session_state.lab.MicrostructureViewer()
            
            with st.spinner("Generating 3D microstructure..."):
                fig = microstructure_viewer.visualize_microstructure_3d(grain_size=grain_size)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if st.button("Generate EBSD Patterns", type="primary"):
            microstructure_viewer = st.session_state.lab.MicrostructureViewer()
            
            with st.spinner("Generating synthetic EBSD patterns..."):
                fig = microstructure_viewer.ebsd_simulation(grain_size=grain_size)
                st.plotly_chart(fig, use_container_width=True)

elif page == "🧪 Alloy Designer":
    st.markdown('<h2 class="section-header">Alloy Design Studio</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="material-card">
    Design your custom alloy by specifying composition and target properties.
    The system uses empirical strengthening models to predict mechanical properties.
    </div>
    """, unsafe_allow_html=True)
    
    # Composition input
    st.markdown('<h3 class="subsection-header">Alloy Composition (wt%)</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    alloying_elements = {}
    
    with col1:
        C = st.number_input("Carbon (C)", 0.0, 2.0, 0.35, 0.01)
        if C > 0:
            alloying_elements["C"] = C
    
    with col2:
        Mn = st.number_input("Manganese (Mn)", 0.0, 5.0, 1.0, 0.1)
        if Mn > 0:
            alloying_elements["Mn"] = Mn
    
    with col3:
        Si = st.number_input("Silicon (Si)", 0.0, 3.0, 0.3, 0.1)
        if Si > 0:
            alloying_elements["Si"] = Si
    
    with col4:
        Cr = st.number_input("Chromium (Cr)", 0.0, 20.0, 1.5, 0.1)
        if Cr > 0:
            alloying_elements["Cr"] = Cr
    
    # Additional elements
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        Ni = st.number_input("Nickel (Ni)", 0.0, 20.0, 0.0, 0.1)
        if Ni > 0:
            alloying_elements["Ni"] = Ni
    
    with col2:
        Mo = st.number_input("Molybdenum (Mo)", 0.0, 5.0, 0.25, 0.01)
        if Mo > 0:
            alloying_elements["Mo"] = Mo
    
    with col3:
        V = st.number_input("Vanadium (V)", 0.0, 2.0, 0.0, 0.01)
        if V > 0:
            alloying_elements["V"] = V
    
    with col4:
        Ti = st.number_input("Titanium (Ti)", 0.0, 2.0, 0.0, 0.01)
        if Ti > 0:
            alloying_elements["Ti"] = Ti
    
    # Target properties
    st.markdown('<h3 class="subsection-header">Target Properties</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_yield = st.number_input("Target Yield Strength (MPa)", 100.0, 2000.0, 800.0, 10.0)
    
    with col2:
        target_elongation = st.number_input("Target Elongation (%)", 1.0, 50.0, 12.0, 0.5)
    
    if st.button("Design Alloy", type="primary"):
        if not alloying_elements:
            st.error("Please specify at least one alloying element!")
        else:
            with st.spinner("Calculating alloy properties..."):
                alloy_result = st.session_state.lab.design_alloy(
                    base_element="Fe",
                    alloying_elements=alloying_elements,
                    target_properties={
                        "yield_strength": target_yield,
                        "elongation": target_elongation
                    }
                )
                
                st.session_state.test_results['alloy'] = alloy_result
                
                # Display results
                st.markdown('<h3 class="subsection-header">Designed Alloy Properties</h3>', unsafe_allow_html=True)
                
                # Main properties
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Yield Strength", f"{alloy_result['predicted_yield_strength']:.0f} MPa")
                
                with col2:
                    st.metric("Tensile Strength", f"{alloy_result['predicted_tensile_strength']:.0f} MPa")
                
                with col3:
                    st.metric("Elongation", f"{alloy_result['predicted_elongation']:.1f} %")
                
                with col4:
                    st.metric("Young's Modulus", f"{alloy_result['predicted_youngs_modulus']:.0f} GPa")
                
                # Strengthening contributions
                st.markdown('<h3 class="subsection-header">Strengthening Contributions</h3>', unsafe_allow_html=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        name='Solid Solution',
                        x=['Solid Solution', 'Precipitation', 'Grain Boundary'],
                        y=[alloy_result['solid_solution_contribution'],
                           alloy_result['precipitation_contribution'],
                           alloy_result['grain_boundary_contribution']]
                    )
                ])
                
                fig.update_layout(
                    title="Strengthening Mechanism Contributions",
                    yaxis_title="Strength Contribution (MPa)",
                    height=400,
                    template=theme
                )
                
                st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Data Export":
    st.markdown('<h2 class="section-header">Data Export & Certification</h2>', unsafe_allow_html=True)
    
    if not st.session_state.test_results:
        st.info("No test results available for export. Run some tests first!")
        st.stop()
    
    # Generate test certificate
    st.markdown('<h3 class="subsection-header">Generate Test Certificate</h3>', unsafe_allow_html=True)
    
    test_type = st.selectbox(
        "Test Type",
        ["Tensile Test", "Fatigue Test", "Fracture Test", "Creep Test", "Alloy Design"]
    )
    
    material_used = st.session_state.current_material or "Custom Alloy"
    
    if st.button("Generate ISO Certificate", type="primary"):
        if test_type in ["Tensile Test", "Alloy Design"] and test_type in st.session_state.test_results:
            certificate = st.session_state.lab.generate_test_certificate(
                test_type=test_type,
                material=material_used,
                properties=st.session_state.test_results.get(test_type.lower().replace(" ", "_"), {})
            )
            
            st.session_state.test_results['certificate'] = certificate
            
            # Display certificate
            st.markdown("---")
            st.markdown('<div class="material-card">', unsafe_allow_html=True)
            st.markdown("### 📜 ISO Test Certificate")
            st.markdown("---")
            
            for key, value in certificate.items():
                if isinstance(value, dict):
                    st.markdown(f"**{key.replace('_', ' ').title()}:**")
                    for k, v in value.items():
                        st.markdown(f"  - **{k.replace('_', ' ').title()}:** {v}")
                else:
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Export options
            st.markdown('<h3 class="subsection-header">Export Options</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_data = json.dumps(certificate, indent=2, default=str)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name="test_certificate.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV export for test data
                if 'tensile' in st.session_state.test_results:
                    df_tensile = pd.DataFrame([st.session_state.test_results['tensile']])
                    csv_data = df_tensile.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name="test_data.csv",
                        mime="text/csv"
                    )
    
    # Export all test data
    st.markdown("---")
    st.markdown('<h3 class="subsection-header">Export All Test Data</h3>', unsafe_allow_html=True)
    
    if st.button("Export Complete Dataset"):
        # Create comprehensive dataset
        all_data = {
            "material": material_used,
            "test_results": st.session_state.test_results
        }
        
        if st.session_state.current_microstructure:
            all_data["microstructure"] = {
                "grain_size": st.session_state.current_microstructure.grain_size,
                "phase_fractions": st.session_state.current_microstructure.phase_fractions,
                "crystal_structure": st.session_state.current_microstructure.crystal_structure.value
            }
        
        # Export as JSON
        json_all = json.dumps(all_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download All Data (JSON)",
            data=json_all,
            file_name="complete_lab_data.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9rem; padding: 1rem;'>
    <b>Virtual Materials Testing Laboratory v3.0</b><br>
    Academic Edition | ISO 6892-1 Compliant | Multi-scale Modeling Framework<br>
    © 2024 Materials Science Simulation Platform
</div>
""", unsafe_allow_html=True)
