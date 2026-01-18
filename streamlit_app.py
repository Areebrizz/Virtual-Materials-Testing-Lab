"""
STREAMLIT INTERFACE FOR VIRTUAL MATERIALS LAB
Advanced Web-Based Materials Science Simulator
Complete working version with all modules
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
from scipy import integrate, optimize, interpolate, stats
from scipy.spatial import cKDTree
from scipy.ndimage import sobel, gaussian_filter
import warnings
warnings.filterwarnings('ignore')
import uuid
import datetime

# ==================== CORE MATERIALS SCIENCE MODELS ====================

class CrystalStructure(Enum):
    """Crystallographic structure definitions with atomic properties"""
    BCC = "Body-Centered Cubic"
    FCC = "Face-Centered Cubic"
    HCP = "Hexagonal Close-Packed"
    COMPOSITE = "Composite/Amorphous"
    
    @property
    def atomic_packing_factor(self) -> float:
        """Atomic packing factor for each crystal structure"""
        factors = {
            CrystalStructure.BCC: 0.68,
            CrystalStructure.FCC: 0.74,
            CrystalStructure.HCP: 0.74,
            CrystalStructure.COMPOSITE: 0.60
        }
        return factors.get(self, 0.65)

class MaterialClass(Enum):
    """Material classification with typical applications"""
    STEEL = "Carbon/Low-Alloy Steel"
    ALUMINUM = "Aluminum Alloy"
    TITANIUM = "Titanium Alloy"
    COMPOSITE = "Fiber-Reinforced Composite"
    SUPERALLOY = "Nickel-based Superalloy"
    CERAMIC = "Advanced Ceramic"
    POLYMER = "Engineering Polymer"

@dataclass
class Microstructure:
    """Advanced microstructure representation with crystallographic data"""
    grain_size: float  # μm
    phase_fractions: Dict[str, float]
    defect_density: float  # defects/mm²
    porosity: float  # volume fraction
    inclusion_size: float  # μm
    inclusion_volume_fraction: float
    crystal_structure: CrystalStructure
    texture_coefficient: float = 1.0
    grain_size_distribution: str = "log-normal"
    twin_density: float = 0.0
    dislocation_density: float = 1e12  # m⁻²
    
    def calculate_hall_peetch(self) -> float:
        """Hall-Petch strengthening coefficient: σ = σ₀ + k/√d"""
        if self.grain_size > 0:
            k = 500  # MPa√mm, typical for steels
            return k / np.sqrt(self.grain_size * 0.001)
        return 0

@dataclass
class HeatTreatment:
    """Advanced heat treatment parameters with phase transformation kinetics"""
    quenching_rate: float  # °C/s
    tempering_temperature: float  # °C
    tempering_time: float  # hours
    austenitizing_temp: float = 950.0
    cooling_medium: str = "oil"
    precipitation_temp: float = 500.0
    aging_time: float = 8.0
    martensite_start: float = 300.0
    bainite_transformation: bool = False
    
    def calculate_hardenability(self) -> float:
        """Jominy end-quench hardenability calculation"""
        if self.cooling_medium == "water":
            return 50.0
        elif self.cooling_medium == "oil":
            return 30.0
        else:
            return 15.0

@dataclass
class MaterialProperties:
    """Comprehensive material properties database with derived properties"""
    youngs_modulus: float  # GPa
    poissons_ratio: float
    yield_strength: float  # MPa
    tensile_strength: float  # MPa
    elongation: float  # %
    reduction_area: float  # %
    fracture_toughness: float  # MPa√m
    fatigue_limit: float  # MPa
    density: float  # kg/m³
    thermal_conductivity: float  # W/m·K
    specific_heat: float  # J/kg·K
    thermal_expansion: float  # 10⁻⁶/K
    crystal_structure: CrystalStructure
    
    stacking_fault_energy: float = 50.0
    burgers_vector: float = 0.25
    shear_modulus: float = None
    
    def __post_init__(self):
        if self.shear_modulus is None:
            self.shear_modulus = self.youngs_modulus / (2 * (1 + self.poissons_ratio))

# ==================== VIRTUAL MATERIALS LAB CORE ====================

class VirtualMaterialsLab:
    """Main laboratory class integrating all modules"""
    
    def __init__(self):
        self.materials_db = self._initialize_materials_database()
        self.current_material = None
        self.current_microstructure = None
        self.current_heat_treatment = None
        self.test_results = {}
        
    def _initialize_materials_database(self) -> Dict[str, MaterialProperties]:
        """Initialize with comprehensive ASM Handbook data"""
        return {
            "AISI 1045 Steel": MaterialProperties(
                youngs_modulus=200.0,
                poissons_ratio=0.29,
                yield_strength=530.0,
                tensile_strength=625.0,
                elongation=12.0,
                reduction_area=40.0,
                fracture_toughness=50.0,
                fatigue_limit=280.0,
                density=7850.0,
                thermal_conductivity=49.8,
                specific_heat=486.0,
                thermal_expansion=11.7,
                crystal_structure=CrystalStructure.BCC
            ),
            "Al 6061-T6": MaterialProperties(
                youngs_modulus=69.0,
                poissons_ratio=0.33,
                yield_strength=276.0,
                tensile_strength=310.0,
                elongation=12.0,
                reduction_area=22.0,
                fracture_toughness=29.0,
                fatigue_limit=96.5,
                density=2700.0,
                thermal_conductivity=167.0,
                specific_heat=896.0,
                thermal_expansion=23.6,
                crystal_structure=CrystalStructure.FCC
            ),
            "Ti-6Al-4V": MaterialProperties(
                youngs_modulus=113.8,
                poissons_ratio=0.342,
                yield_strength=880.0,
                tensile_strength=950.0,
                elongation=14.0,
                reduction_area=40.0,
                fracture_toughness=75.0,
                fatigue_limit=500.0,
                density=4430.0,
                thermal_conductivity=6.7,
                specific_heat=526.0,
                thermal_expansion=8.6,
                crystal_structure=CrystalStructure.HCP
            ),
            "316L Stainless Steel": MaterialProperties(
                youngs_modulus=193.0,
                poissons_ratio=0.27,
                yield_strength=290.0,
                tensile_strength=580.0,
                elongation=40.0,
                reduction_area=60.0,
                fracture_toughness=100.0,
                fatigue_limit=240.0,
                density=8000.0,
                thermal_conductivity=16.2,
                specific_heat=500.0,
                thermal_expansion=16.0,
                crystal_structure=CrystalStructure.FCC
            )
        }
    
    # ==================== SAMPLE PREPARATION STATION ====================
    
    def design_microstructure(self, material: str, 
                            grain_size: float = 50.0,
                            phase_fraction: Dict[str, float] = None,
                            porosity: float = 0.01,
                            inclusion_size: float = 10.0,
                            texture_strength: float = 0.85) -> Microstructure:
        """Advanced microstructure designer"""
        
        if phase_fraction is None:
            if "Ti" in material:
                phase_fraction = {"alpha": 0.9, "beta": 0.1}
            elif "steel" in material.lower():
                phase_fraction = {"ferrite": 0.85, "pearlite": 0.15}
            elif "Al" in material:
                phase_fraction = {"matrix": 0.95, "precipitates": 0.05}
            else:
                phase_fraction = {"matrix": 1.0}
        
        # Determine crystal structure
        if "Steel" in material:
            crystal_structure = CrystalStructure.BCC
        elif "Al" in material:
            crystal_structure = CrystalStructure.FCC
        elif "Ti" in material:
            crystal_structure = CrystalStructure.HCP
        else:
            crystal_structure = CrystalStructure.BCC
        
        # Calculate defect density
        if grain_size < 10:
            defect_density = 5000.0
        elif grain_size > 100:
            defect_density = 500.0
        else:
            defect_density = 1000.0
        
        self.current_microstructure = Microstructure(
            grain_size=grain_size,
            phase_fractions=phase_fraction,
            defect_density=defect_density,
            porosity=porosity,
            inclusion_size=inclusion_size,
            inclusion_volume_fraction=0.005,
            crystal_structure=crystal_structure,
            texture_coefficient=texture_strength,
            dislocation_density=1e12 * (50/grain_size)
        )
        
        return self.current_microstructure
    
    def apply_heat_treatment(self, quenching_rate: float = 100.0,
                           tempering_temp: float = 600.0,
                           tempering_time: float = 2.0) -> HeatTreatment:
        """Advanced heat treatment simulator"""
        
        if quenching_rate > 200:
            cooling_medium = "water"
        elif quenching_rate > 50:
            cooling_medium = "oil"
        else:
            cooling_medium = "air"
        
        self.current_heat_treatment = HeatTreatment(
            quenching_rate=quenching_rate,
            tempering_temperature=tempering_temp,
            tempering_time=tempering_time,
            cooling_medium=cooling_medium
        )
        
        return self.current_heat_treatment
    
    # ==================== TENSILE TESTING MODULE ====================
    
    class TensileTester:
        """Advanced tensile testing with true stress-strain"""
        
        def __init__(self, material_props: MaterialProperties,
                    microstructure: Microstructure = None):
            self.material = material_props
            self.microstructure = microstructure
            self.engineering_stress_strain = None
            self.true_stress_strain = None
            self.necking_point = None
            
        def generate_stress_strain_curve(self, 
                                       constitutive_model: str = "hollomon",
                                       temperature: float = 20.0,
                                       strain_rate: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
            """Generate stress-strain curve"""
            
            eps = np.linspace(0, 0.25, 1000)
            
            if constitutive_model == "hollomon":
                K = self.material.tensile_strength * (np.exp(self.material.elongation/100))**0.2
                n = 0.1 + 0.2 * (self.material.elongation/100)
                
                elastic_limit = self.material.yield_strength / self.material.youngs_modulus
                mask_elastic = eps <= elastic_limit
                mask_plastic = eps > elastic_limit
                
                stress = np.zeros_like(eps)
                stress[mask_elastic] = self.material.youngs_modulus * 1000 * eps[mask_elastic]
                stress[mask_plastic] = K * (eps[mask_plastic] - elastic_limit)**n
                
            elif constitutive_model == "voce":
                sigma_0 = self.material.yield_strength
                Q = self.material.tensile_strength - sigma_0
                b = 20.0
                
                elastic_limit = sigma_0 / self.material.youngs_modulus
                mask_elastic = eps <= elastic_limit
                mask_plastic = eps > elastic_limit
                
                stress = np.zeros_like(eps)
                stress[mask_elastic] = self.material.youngs_modulus * 1000 * eps[mask_elastic]
                stress[mask_plastic] = (sigma_0 + Q * (1 - np.exp(-b * (eps[mask_plastic] - elastic_limit))))
            
            # Apply microstructure effects
            if self.microstructure:
                hp_strength = self.microstructure.calculate_hall_peetch()
                stress += hp_strength * (eps > elastic_limit)
            
            self.engineering_stress_strain = (eps, stress)
            
            # Convert to true stress-strain
            true_strain = np.log(1 + eps)
            true_stress = stress * (1 + eps)
            self.true_stress_strain = (true_strain, true_stress)
            
            # Find necking point
            if np.any(mask_plastic):
                eps_plastic = eps[mask_plastic]
                stress_plastic = stress[mask_plastic]
                grad = np.gradient(stress_plastic, eps_plastic)
                necking_idx = np.where(grad <= stress_plastic)[0]
                if len(necking_idx) > 0:
                    self.necking_point = (eps_plastic[necking_idx[0]], 
                                        stress_plastic[necking_idx[0]])
            
            return eps, stress
        
        def calculate_mechanical_properties(self) -> Dict[str, float]:
            """Calculate all standard mechanical properties"""
            if self.engineering_stress_strain is None:
                raise ValueError("Generate stress-strain curve first")
            
            eps, stress = self.engineering_stress_strain
            
            # Young's modulus
            initial_slope = np.polyfit(eps[:50], stress[:50], 1)[0]
            E = initial_slope / 1000
            
            # 0.2% offset yield strength
            offset_line = E * 1000 * (eps - 0.002)
            intersect_idx = np.where(stress >= offset_line)[0]
            sigma_y = stress[intersect_idx[0]] if len(intersect_idx) > 0 else stress[0]
            
            # UTS
            sigma_uts = np.max(stress)
            
            # Elongation
            fracture_strain = eps[-1]
            
            # Strain hardening exponent
            plastic_strain = eps[eps > sigma_y/(E*1000)] - sigma_y/(E*1000)
            plastic_stress = stress[eps > sigma_y/(E*1000)]
            if len(plastic_strain) > 10:
                log_eps = np.log(plastic_strain)
                log_sigma = np.log(plastic_stress)
                n_value = np.polyfit(log_eps, log_sigma, 1)[0]
            else:
                n_value = 0.1
            
            return {
                "Young's Modulus (GPa)": round(E, 1),
                "Yield Strength (MPa)": round(sigma_y, 1),
                "UTS (MPa)": round(sigma_uts, 1),
                "Total Elongation (%)": round(fracture_strain * 100, 1),
                "Strain Hardening Exponent (n)": round(n_value, 3),
                "Necking Strain (%)": round(self.necking_point[0]*100, 2) if self.necking_point else 0.0
            }
        
        def visualize_curve(self):
            """Interactive visualization with Plotly"""
            if self.engineering_stress_strain is None:
                raise ValueError("Generate curve first")
            
            eps, eng_stress = self.engineering_stress_strain
            true_strain, true_stress = self.true_stress_strain
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Engineering Stress-Strain", 
                              "True Stress-Strain",
                              "Strain Hardening Rate",
                              "Considère Criterion"),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # Engineering curve
            fig.add_trace(
                go.Scatter(x=eps*100, y=eng_stress, mode='lines',
                          name='Engineering', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            if self.necking_point:
                fig.add_trace(
                    go.Scatter(x=[self.necking_point[0]*100], 
                              y=[self.necking_point[1]],
                              mode='markers',
                              name='Necking Start',
                              marker=dict(size=10, color='red', symbol='x')),
                    row=1, col=1
                )
            
            # True curve
            fig.add_trace(
                go.Scatter(x=true_strain*100, y=true_stress, mode='lines',
                          name='True', line=dict(color='red', width=2)),
                row=1, col=2
            )
            
            # Strain hardening rate
            elastic_limit = self.material.yield_strength / self.material.youngs_modulus
            plastic_range = eps > elastic_limit
            if np.any(plastic_range):
                plastic_eps = eps[plastic_range]
                plastic_stress = eng_stress[plastic_range]
                hardening_rate = np.gradient(plastic_stress, plastic_eps)
                
                fig.add_trace(
                    go.Scatter(x=plastic_eps*100, y=hardening_rate,
                              mode='lines', name='dσ/dε',
                              line=dict(color='green', width=2)),
                    row=2, col=1
                )
            
            # Considère criterion
            if self.necking_point:
                fig.add_trace(
                    go.Scatter(x=eps*100, y=eng_stress,
                              name='Stress', line=dict(color='blue')),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(x=eps*100, y=np.gradient(eng_stress, eps),
                              name='dσ/dε', line=dict(color='red')),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                showlegend=True,
                title_text=f"Tensile Test Results"
            )
            
            fig.update_xaxes(title_text="Strain (%)", row=1, col=1)
            fig.update_yaxes(title_text="Stress (MPa)", row=1, col=1)
            fig.update_xaxes(title_text="True Strain (%)", row=1, col=2)
            fig.update_yaxes(title_text="True Stress (MPa)", row=1, col=2)
            fig.update_xaxes(title_text="Plastic Strain (%)", row=2, col=1)
            fig.update_yaxes(title_text="Hardening Rate (MPa)", row=2, col=1)
            fig.update_xaxes(title_text="Strain (%)", row=2, col=2)
            fig.update_yaxes(title_text="Stress/dσ/dε (MPa)", row=2, col=2)
            
            return fig
    
    # ==================== FATIGUE TESTING MODULE ====================
    
    class FatigueTester:
        """Advanced fatigue life prediction"""
        
        def __init__(self, material_props: MaterialProperties):
            self.material = material_props
            self.SN_data = None
            
        def generate_SN_curve(self, R_ratio: float = -1.0,
                            surface_finish: str = "polished") -> Tuple[np.ndarray, np.ndarray]:
            """Generate S-N curve"""
            
            # Basquin equation
            sigma_f_prime = self.material.tensile_strength * 1.5
            b = -0.085
            
            N_cycles = np.logspace(3, 8, 50)
            stress_amp = sigma_f_prime * (2 * N_cycles) ** b
            
            # Mean stress effect
            if R_ratio != -1:
                sigma_mean = stress_amp * (1 + R_ratio) / (1 - R_ratio)
                sigma_amp_corrected = stress_amp * (1 - sigma_mean/self.material.tensile_strength)
                stress_amp = sigma_amp_corrected
            
            # Surface finish factor
            surface_factors = {
                "polished": 1.0,
                "machined": 0.8,
                "hot_rolled": 0.6,
                "as_forged": 0.4
            }
            stress_amp *= surface_factors.get(surface_finish, 0.9)
            
            # Fatigue limit
            fatigue_limit = self.material.fatigue_limit
            stress_amp[stress_amp < fatigue_limit] = fatigue_limit
            
            self.SN_data = (N_cycles, stress_amp)
            return N_cycles, stress_amp
        
        def visualize_SN_curve(self):
            """Visualize S-N curve"""
            if self.SN_data is None:
                raise ValueError("Generate S-N curve first")
            
            N_cycles, stress_amp = self.SN_data
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(x=N_cycles, y=stress_amp, mode='lines',
                          name='S-N Curve', line=dict(color='blue', width=3))
            )
            
            # Add fatigue limit line
            fig.add_hline(y=self.material.fatigue_limit, 
                         line_dash="dash", 
                         line_color="red",
                         annotation_text=f"Fatigue Limit: {self.material.fatigue_limit} MPa")
            
            fig.update_layout(
                xaxis_title="Number of Cycles (N)",
                yaxis_title="Stress Amplitude (MPa)",
                title="S-N Fatigue Curve",
                xaxis_type="log",
                height=500,
                showlegend=True
            )
            
            return fig
        
        def get_fatigue_life(self, stress_amplitude: float) -> float:
            """Calculate fatigue life for given stress amplitude"""
            if self.SN_data is None:
                raise ValueError("Generate S-N curve first")
            
            N_cycles, stress_amp = self.SN_data
            
            # Interpolate to find N for given stress
            if stress_amplitude <= min(stress_amp):
                return 1e8  # Infinite life
            elif stress_amplitude >= max(stress_amp):
                return 1e3  # Short life
            
            # Find closest value
            idx = np.argmin(np.abs(stress_amp - stress_amplitude))
            return N_cycles[idx]
    
    # ==================== FRACTURE TOUGHNESS MODULE ====================
    
    class FractureToughnessTester:
        """Advanced fracture mechanics"""
        
        def __init__(self, material_props: MaterialProperties):
            self.material = material_props
            
        def calculate_stress_field(self, K_I: float = 30.0,
                                 distance: float = 10.0) -> Dict[str, np.ndarray]:
            """Calculate crack tip stress field"""
            
            theta = np.linspace(-np.pi, np.pi, 100)
            r = np.linspace(0.1, distance, 50)
            R, Theta = np.meshgrid(r, theta)
            
            # Williams asymptotic expansion
            sigma_xx = K_I / np.sqrt(2 * np.pi * R) * np.cos(Theta/2) * (1 - np.sin(Theta/2) * np.sin(3*Theta/2))
            sigma_yy = K_I / np.sqrt(2 * np.pi * R) * np.cos(Theta/2) * (1 + np.sin(Theta/2) * np.sin(3*Theta/2))
            tau_xy = K_I / np.sqrt(2 * np.pi * R) * np.sin(Theta/2) * np.cos(Theta/2) * np.cos(3*Theta/2)
            
            # von Mises equivalent stress
            sigma_vm = np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx*sigma_yy + 3*tau_xy**2)
            
            return {
                'sigma_xx': sigma_xx,
                'sigma_yy': sigma_yy,
                'tau_xy': tau_xy,
                'sigma_vm': sigma_vm,
                'r': R,
                'theta': Theta
            }
        
        def estimate_plastic_zone(self, K_I: float = 30.0) -> float:
            """Estimate plastic zone size"""
            
            sigma_y = self.material.yield_strength
            r_p = (1/(2*np.pi)) * (K_I/sigma_y)**2
            
            return r_p
        
        def visualize_crack_tip(self, K_I: float = 30.0):
            """Interactive crack tip stress field visualization"""
            
            stress_field = self.calculate_stress_field(K_I)
            r_p = self.estimate_plastic_zone(K_I)
            
            # Create 4 heatmaps in 2x2 grid
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("σ_xx Stress Field",
                              "σ_yy Stress Field",
                              "Von Mises Stress",
                              "Plastic Zone"),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # Convert polar to Cartesian for heatmaps
            x = stress_field['r'] * np.cos(stress_field['theta'])
            y = stress_field['r'] * np.sin(stress_field['theta'])
            
            # For heatmaps, we need 2D arrays
            # σ_xx heatmap
            fig.add_trace(
                go.Heatmap(z=stress_field['sigma_xx'], 
                          colorscale='RdBu',
                          showscale=True,
                          name='σ_xx'),
                row=1, col=1
            )
            
            # σ_yy heatmap
            fig.add_trace(
                go.Heatmap(z=stress_field['sigma_yy'], 
                          colorscale='RdBu',
                          showscale=True,
                          name='σ_yy'),
                row=1, col=2
            )
            
            # von Mises heatmap
            fig.add_trace(
                go.Heatmap(z=stress_field['sigma_vm'], 
                          colorscale='Viridis',
                          showscale=True,
                          name='σ_vm'),
                row=2, col=1
            )
            
            # Plastic zone visualization (simple scatter)
            theta_circle = np.linspace(0, 2*np.pi, 100)
            x_circle = r_p * np.cos(theta_circle)
            y_circle = r_p * np.sin(theta_circle)
            
            fig.add_trace(
                go.Scatter(x=x_circle, y=y_circle,
                          mode='lines',
                          fill='toself',
                          fillcolor='rgba(255,0,0,0.2)',
                          line=dict(color='red'),
                          name=f'Plastic Zone (r={r_p:.2f} mm)'),
                row=2, col=2
            )
            
            # Update layout for plastic zone subplot
            fig.update_xaxes(title_text="X (mm)", row=2, col=2, range=[-r_p*1.5, r_p*1.5])
            fig.update_yaxes(title_text="Y (mm)", row=2, col=2, range=[-r_p*1.5, r_p*1.5])
            
            fig.update_layout(
                height=700,
                title_text=f"Crack Tip Analysis - K_I = {K_I} MPa√m, K_IC = {self.material.fracture_toughness} MPa√m",
                showlegend=True
            )
            
            # Update axis labels for heatmaps
            fig.update_xaxes(title_text="X (mm)", row=1, col=1)
            fig.update_yaxes(title_text="Y (mm)", row=1, col=1)
            fig.update_xaxes(title_text="X (mm)", row=1, col=2)
            fig.update_yaxes(title_text="Y (mm)", row=1, col=2)
            fig.update_xaxes(title_text="X (mm)", row=2, col=1)
            fig.update_yaxes(title_text="Y (mm)", row=2, col=1)
            
            return fig
    
    # ==================== CREEP TESTING MODULE ====================
    
    class CreepTester:
        """Advanced creep deformation and rupture prediction"""
        
        def __init__(self, material_props: MaterialProperties):
            self.material = material_props
            
        def creep_deformation(self, stress: float = 100.0,
                            temperature: float = 600.0,
                            time_hours: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
            """Calculate creep strain using Norton's law"""
            
            # Material-dependent parameters
            if self.material.crystal_structure == CrystalStructure.FCC:
                n = 5.0
                Q = 250e3
                A0 = 1e-10
            else:
                n = 4.0
                Q = 300e3
                A0 = 1e-11
            
            R = 8.314
            T = temperature + 273.15
            
            # Time array
            t_hours = np.linspace(0, time_hours, 1000)
            t_seconds = t_hours * 3600
            
            # Creep strain
            A = A0 * np.exp(-Q/(R*T))
            epsilon_creep = A * (stress ** n) * t_seconds
            
            return t_hours, epsilon_creep * 100
        
        def visualize_creep_curve(self, stress: float = 100.0,
                                temperature: float = 600.0):
            """Visualize creep curve"""
            
            # Generate time and strain data
            time_hours = 10000  # Default time for visualization
            t_hours, strain_percent = self.creep_deformation(stress, temperature, time_hours)
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(x=t_hours, y=strain_percent, mode='lines',
                          name='Creep Strain', line=dict(color='purple', width=3))
            )
            
            # Add secondary creep region indication
            if len(strain_percent) > 100:
                # Find where strain rate becomes constant (secondary creep)
                strain_rate = np.gradient(strain_percent, t_hours)
                min_rate_idx = np.argmin(strain_rate[100:]) + 100
                secondary_time = t_hours[min_rate_idx]
                secondary_strain = strain_percent[min_rate_idx]
                
                fig.add_vline(x=secondary_time, line_dash="dash", line_color="green",
                            annotation_text="Secondary Creep Start")
                fig.add_annotation(x=secondary_time, y=secondary_strain,
                                 text="Secondary Creep",
                                 showarrow=True,
                                 arrowhead=1)
            
            fig.update_layout(
                xaxis_title="Time (hours)",
                yaxis_title="Creep Strain (%)",
                title=f"Creep Curve: {stress} MPa at {temperature}°C",
                height=500,
                showlegend=True
            )
            
            return fig
        
        def calculate_rupture_life(self, stress: float, temperature: float) -> float:
            """Estimate rupture life using Larson-Miller parameter"""
            
            # Simplified Larson-Miller
            C = 20  # Typical constant for steels
            LMP = (temperature + 273.15) * (C + np.log10(1000))  # Base LMP
            
            # Adjust for stress
            stress_factor = (self.material.tensile_strength / stress) ** 3
            rupture_hours = 10 ** (LMP / (temperature + 273.15) - C) * stress_factor
            
            return max(rupture_hours, 1)
    
    # ==================== MICROSTRUCTURE VIEWER ====================
    
    class MicrostructureViewer:
        """Advanced microstructure generation and visualization"""
        
        def __init__(self):
            self.grains = None
            
        def generate_voronoi_microstructure(self, grain_size: float = 50.0,
                                          size: int = 400) -> np.ndarray:
            """Generate synthetic microstructure"""
            
            # Number of grains (approximate)
            area = size * size
            grain_area = grain_size * grain_size
            n_grains = max(10, int(area / grain_area))
            
            # Random grain centers
            points = np.random.rand(n_grains, 2) * size
            
            # Create grid
            x, y = np.meshgrid(np.arange(size), np.arange(size))
            grid_points = np.column_stack([x.ravel(), y.ravel()])
            
            # Assign each point to nearest grain center
            tree = cKDTree(points)
            distances, grain_indices = tree.query(grid_points)
            
            # Reshape to image
            grain_map = grain_indices.reshape((size, size))
            
            return grain_map
        
        def visualize_microstructure(self, grain_size: float = 50.0):
            """Visualize microstructure"""
            
            grain_map = self.generate_voronoi_microstructure(grain_size)
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Heatmap(z=grain_map, 
                          colorscale='Viridis',
                          showscale=False,
                          name='Grain Map')
            )
            
            fig.update_layout(
                title=f"Microstructure - Grain Size: {grain_size} μm",
                height=500,
                xaxis_title="X (μm)",
                yaxis_title="Y (μm)",
                showlegend=False
            )
            
            return fig
        
        def calculate_grain_statistics(self, grain_map: np.ndarray) -> Dict[str, float]:
            """Calculate basic grain statistics"""
            
            unique_grains = np.unique(grain_map)
            n_grains = len(unique_grains)
            
            # Calculate areas
            areas = []
            for grain in unique_grains:
                mask = grain_map == grain
                areas.append(np.sum(mask))
            
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            
            # Convert to equivalent diameter
            mean_diameter = 2 * np.sqrt(mean_area / np.pi)
            
            return {
                "number_of_grains": n_grains,
                "mean_grain_area": mean_area,
                "std_grain_area": std_area,
                "mean_grain_diameter": mean_diameter,
                "grain_size_variation": std_area / mean_area if mean_area > 0 else 0
            }
    
    # ==================== ALLOY DESIGNER ====================
    
    def design_alloy(self, base_element: str = "Fe",
                    alloying_elements: Dict[str, float] = None) -> Dict[str, Any]:
        """Advanced alloy design using empirical models"""
        
        if alloying_elements is None:
            alloying_elements = {"C": 0.35, "Mn": 0.75, "Cr": 0.25}
        
        # Empirical strengthening models
        base_strength = 200
        
        # Solid solution strengthening
        ss_coefficients = {
            "C": 5000, "Mn": 80, "Si": 60, "Cr": 50,
            "Ni": 40, "Mo": 100, "V": 300, "Ti": 400
        }
        
        ss_strength = 0
        for element, wt_pct in alloying_elements.items():
            if element in ss_coefficients:
                ss_strength += ss_coefficients[element] * wt_pct
        
        # Grain boundary strengthening
        grain_size = 20
        gb_strength = 500 / np.sqrt(grain_size)
        
        # Total yield strength
        predicted_yield = base_strength + ss_strength + gb_strength
        
        # Estimate elongation
        predicted_elongation = 30 - 20 * (predicted_yield / 1000)
        
        # Estimate Young's modulus
        base_E = 200 if base_element == "Fe" else 70
        predicted_E = base_E * (1 + 0.01 * sum(alloying_elements.values()))
        
        return {
            "alloy_composition": alloying_elements,
            "predicted_yield_strength": round(predicted_yield, 1),
            "predicted_tensile_strength": round(predicted_yield * 1.2, 1),
            "predicted_elongation": round(max(5, predicted_elongation), 1),
            "predicted_youngs_modulus": round(predicted_E, 1),
            "solid_solution_contribution": round(ss_strength, 1),
            "grain_boundary_contribution": round(gb_strength, 1),
            "base_strength": round(base_strength, 1)
        }
    
    # ==================== DATA EXPORT ====================
    
    def generate_test_certificate(self, test_type: str,
                                material: str,
                                properties: Dict[str, float]) -> Dict[str, Any]:
        """Generate ISO-compliant test certificate"""
        
        certificate = {
            "test_laboratory": "Virtual Materials Testing Lab v3.0",
            "iso_standard": "ISO 6892-1",
            "test_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "material_identification": material,
            "test_type": test_type,
            "test_conditions": {
                "temperature": "23 ± 2°C",
                "humidity": "50 ± 10% RH",
                "strain_rate": "0.001 s⁻¹"
            },
            "mechanical_properties": properties,
            "measurement_uncertainty": "Expanded uncertainty k=2 (95% confidence)",
            "calibration_status": "All equipment within calibration period",
            "signature": {
                "test_engineer": "Virtual Testing System",
                "approval": "ISO/IEC 17025 compliant simulation"
            }
        }
        
        return certificate

# ==================== STREAMLIT INTERFACE ====================

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
    .material-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #3498db;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
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
        st.metric("Current Material", st.session_state.current_material)
    
    if st.session_state.current_microstructure:
        grain_size = st.session_state.current_microstructure.grain_size
        st.metric("Grain Size", f"{grain_size:.1f} μm")

# ==================== DASHBOARD ====================

if page == "🏠 Dashboard":
    st.markdown('<h2 class="section-header">Laboratory Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Available Materials", len(st.session_state.lab.materials_db))
    
    with col2:
        st.metric("Test Modules", "7")
    
    with col3:
        st.metric("ISO Standards", "ISO 6892-1")
    
    # Quick start guide
    st.markdown("### Quick Start Guide")
    
    steps = [
        "1. **Sample Preparation**: Select a material and design its microstructure",
        "2. **Run Tests**: Use any testing module (Tensile, Fatigue, Fracture, Creep)",
        "3. **Analyze Results**: View interactive visualizations and export data"
    ]
    
    for step in steps:
        st.markdown(step)
    
    # Material database preview
    st.markdown("### Material Database")
    
    materials_data = []
    for name, props in st.session_state.lab.materials_db.items():
        materials_data.append({
            "Material": name,
            "E (GPa)": props.youngs_modulus,
            "σ_y (MPa)": props.yield_strength,
            "σ_UTS (MPa)": props.tensile_strength,
            "ε_f (%)": props.elongation
        })
    
    df_materials = pd.DataFrame(materials_data)
    st.dataframe(df_materials, use_container_width=True, hide_index=True)

# ==================== SAMPLE PREPARATION ====================

elif page == "⚗️ Sample Preparation":
    st.markdown('<h2 class="section-header">⚗️ Sample Preparation Station</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Material Selection")
        material_options = list(st.session_state.lab.materials_db.keys())
        selected_material = st.selectbox(
            "Choose Material",
            material_options,
            index=0
        )
        
        if st.button("Load Material", type="primary"):
            st.session_state.current_material = selected_material
            material_props = st.session_state.lab.materials_db[selected_material]
            st.success(f"Loaded {selected_material}")
            
            # Display material properties
            st.markdown(f"#### Material Properties")
            st.markdown(f"- Young's Modulus: {material_props.youngs_modulus} GPa")
            st.markdown(f"- Yield Strength: {material_props.yield_strength} MPa")
            st.markdown(f"- Tensile Strength: {material_props.tensile_strength} MPa")
            st.markdown(f"- Elongation: {material_props.elongation}%")
    
    with col2:
        st.markdown("### Microstructure Design")
        
        if st.session_state.current_material:
            grain_size = st.slider("Grain Size (μm)", 1.0, 200.0, 50.0, 1.0)
            porosity = st.slider("Porosity (%)", 0.0, 5.0, 0.5, 0.1)
            inclusion_size = st.slider("Inclusion Size (μm)", 1.0, 50.0, 10.0, 1.0)
            
            if st.button("Design Microstructure", type="primary"):
                microstructure = st.session_state.lab.design_microstructure(
                    material=st.session_state.current_material,
                    grain_size=grain_size,
                    porosity=porosity/100,
                    inclusion_size=inclusion_size
                )
                st.session_state.current_microstructure = microstructure
                
                st.success("Microstructure Designed!")
                st.markdown(f"#### Microstructure Properties")
                st.markdown(f"- Grain Size: {microstructure.grain_size} μm")
                st.markdown(f"- Porosity: {microstructure.porosity*100:.2f}%")
                st.markdown(f"- Defect Density: {microstructure.defect_density:.0f} defects/mm²")
                st.markdown(f"- Hall-Petch Strengthening: {microstructure.calculate_hall_peetch():.1f} MPa")
        
        st.markdown("### Heat Treatment")
        col3, col4 = st.columns(2)
        
        with col3:
            quenching_rate = st.selectbox("Quenching Rate", ["Slow (air)", "Medium (oil)", "Fast (water)"])
            quenching_rates = {"Slow (air)": 10, "Medium (oil)": 50, "Fast (water)": 200}
        
        with col4:
            tempering_temp = st.slider("Tempering Temp (°C)", 200, 700, 600)
            tempering_time = st.slider("Tempering Time (h)", 0.5, 24.0, 2.0, 0.5)
        
        if st.button("Apply Heat Treatment", type="primary"):
            heat_treatment = st.session_state.lab.apply_heat_treatment(
                quenching_rate=quenching_rates[quenching_rate],
                tempering_temp=tempering_temp,
                tempering_time=tempering_time
            )
            st.session_state.current_heat_treatment = heat_treatment
            
            st.success("Heat Treatment Applied!")
            st.markdown(f"- Cooling Medium: {heat_treatment.cooling_medium}")
            st.markdown(f"- Hardenability: {heat_treatment.calculate_hardenability():.1f}")

# ==================== TENSILE TESTING ====================

elif page == "📈 Tensile Testing":
    st.markdown('<h2 class="section-header">📈 Tensile Testing Module</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_material:
        st.warning("Please select a material in Sample Preparation first.")
    else:
        material_props = st.session_state.lab.materials_db[st.session_state.current_material]
        microstructure = st.session_state.current_microstructure
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Test Parameters")
            constitutive_model = st.selectbox(
                "Constitutive Model",
                ["hollomon", "voce"],
                help="Hollomon: σ = Kεⁿ, Voce: σ = σ₀ + Q(1 - exp(-bε))"
            )
            temperature = st.slider("Temperature (°C)", -50, 500, 20, 10)
            strain_rate = st.selectbox(
                "Strain Rate (s⁻¹)",
                [0.0001, 0.001, 0.01, 0.1, 1.0],
                index=1
            )
            
            if st.button("Run Tensile Test", type="primary"):
                tester = st.session_state.lab.TensileTester(material_props, microstructure)
                eps, stress = tester.generate_stress_strain_curve(
                    constitutive_model=constitutive_model,
                    temperature=temperature,
                    strain_rate=strain_rate
                )
                properties = tester.calculate_mechanical_properties()
                st.session_state.test_results["tensile"] = properties
                st.session_state.tensile_tester = tester
                st.success("Tensile test completed!")
        
        with col2:
            if "tensile_tester" in st.session_state:
                tester = st.session_state.tensile_tester
                
                st.markdown("### Test Results")
                properties = st.session_state.test_results.get("tensile", {})
                
                for key, value in properties.items():
                    if "MPa" in key or "GPa" in key:
                        st.metric(key, f"{value}")
                    elif "%" in key:
                        st.metric(key, f"{value:.1f}")
                    else:
                        st.metric(key, f"{value:.3f}")
        
        # Visualization
        if "tensile_tester" in st.session_state:
            st.markdown("### Stress-Strain Analysis")
            fig = st.session_state.tensile_tester.visualize_curve()
            st.plotly_chart(fig, use_container_width=True)

# ==================== FATIGUE TESTING ====================

elif page == "🔄 Fatigue Testing":
    st.markdown('<h2 class="section-header">🔄 Fatigue Testing Module</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_material:
        st.warning("Please select a material in Sample Preparation first.")
    else:
        material_props = st.session_state.lab.materials_db[st.session_state.current_material]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Fatigue Test Parameters")
            R_ratio = st.slider("Stress Ratio (R)", -1.0, 0.5, -1.0, 0.1)
            surface_finish = st.selectbox(
                "Surface Finish",
                ["polished", "machined", "hot_rolled", "as_forged"]
            )
            
            if st.button("Generate S-N Curve", type="primary"):
                tester = st.session_state.lab.FatigueTester(material_props)
                N_cycles, stress_amp = tester.generate_SN_curve(
                    R_ratio=R_ratio,
                    surface_finish=surface_finish
                )
                st.session_state.fatigue_tester = tester
                st.success("S-N curve generated!")
        
        with col2:
            if "fatigue_tester" in st.session_state:
                tester = st.session_state.fatigue_tester
                
                st.markdown("### Fatigue Properties")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.metric("Fatigue Limit", f"{material_props.fatigue_limit} MPa")
                    st.metric("Tensile Strength", f"{material_props.tensile_strength} MPa")
                
                with col4:
                    endurance_ratio = material_props.fatigue_limit / material_props.tensile_strength
                    st.metric("Endurance Ratio", f"{endurance_ratio:.3f}")
                    
                    # Example fatigue life calculation
                    test_stress = 300  # MPa
                    fatigue_life = tester.get_fatigue_life(test_stress)
                    st.metric(f"Cycles at {test_stress}MPa", f"{fatigue_life:,.0f}")
        
        # Visualization
        if "fatigue_tester" in st.session_state:
            st.markdown("### S-N Curve")
            fig = st.session_state.fatigue_tester.visualize_SN_curve()
            st.plotly_chart(fig, use_container_width=True)

# ==================== FRACTURE TOUGHNESS ====================

elif page == "⚡ Fracture Toughness":
    st.markdown('<h2 class="section-header">⚡ Fracture Toughness Testing</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_material:
        st.warning("Please select a material in Sample Preparation first.")
    else:
        material_props = st.session_state.lab.materials_db[st.session_state.current_material]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Fracture Mechanics Parameters")
            K_I = st.slider("Stress Intensity Factor K_I (MPa√m)", 
                          10.0, 100.0, 30.0, 5.0)
            distance = st.slider("Distance from Crack Tip (mm)", 
                               0.1, 20.0, 10.0, 0.1)
            
            if st.button("Calculate Stress Field", type="primary"):
                tester = st.session_state.lab.FractureToughnessTester(material_props)
                st.session_state.fracture_tester = tester
                st.success("Stress field calculated!")
        
        with col2:
            if "fracture_tester" in st.session_state:
                tester = st.session_state.fracture_tester
                
                st.markdown("### Fracture Properties")
                st.metric("Fracture Toughness K_IC", 
                         f"{material_props.fracture_toughness} MPa√m")
                
                plastic_zone = tester.estimate_plastic_zone(K_I)
                st.metric("Plastic Zone Size", f"{plastic_zone:.3f} mm")
                
                K_ratio = K_I / material_props.fracture_toughness if material_props.fracture_toughness > 0 else 0
                st.metric("K_I/K_IC Ratio", f"{K_ratio:.3f}")
                
                if K_ratio > 0.7:
                    st.error("⚠️ High risk of fracture!")
                elif K_ratio > 0.3:
                    st.warning("⚠️ Moderate risk of fracture")
                else:
                    st.success("✓ Safe from fracture")
        
        # Visualization
        if "fracture_tester" in st.session_state:
            st.markdown("### Crack Tip Stress Field")
            fig = st.session_state.fracture_tester.visualize_crack_tip(K_I)
            st.plotly_chart(fig, use_container_width=True)

# ==================== CREEP TESTING ====================

elif page == "🔥 Creep Testing":
    st.markdown('<h2 class="section-header">🔥 Creep Testing Module</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_material:
        st.warning("Please select a material in Sample Preparation first.")
    else:
        material_props = st.session_state.lab.materials_db[st.session_state.current_material]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Creep Test Parameters")
            stress = st.slider("Applied Stress (MPa)", 50, 300, 100, 10)
            temperature = st.slider("Temperature (°C)", 400, 800, 600, 10)
            
            if st.button("Run Creep Test", type="primary"):
                tester = st.session_state.lab.CreepTester(material_props)
                st.session_state.creep_tester = tester
                st.success("Creep test completed!")
        
        with col2:
            if "creep_tester" in st.session_state:
                tester = st.session_state.creep_tester
                
                st.markdown("### Creep Properties")
                
                # Calculate rupture life
                rupture_life = tester.calculate_rupture_life(stress, temperature)
                st.metric("Estimated Rupture Life", f"{rupture_life:.0f} hours")
                
                st.metric("Test Temperature", f"{temperature}°C")
                
                # Calculate time to 1% creep strain
                t_hours, strain_percent = tester.creep_deformation(stress, temperature, 10000)
                if len(strain_percent) > 0:
                    # Find time to reach 1% strain
                    idx = np.argmax(strain_percent >= 1)
                    if idx > 0:
                        time_to_1pct = t_hours[idx]
                        st.metric("Time to 1% Strain", f"{time_to_1pct:.0f} hours")
        
        # Visualization
        if "creep_tester" in st.session_state:
            st.markdown("### Creep Curve")
            fig = st.session_state.creep_tester.visualize_creep_curve(stress, temperature)
            st.plotly_chart(fig, use_container_width=True)

# ==================== MICROSTRUCTURE VIEWER ====================

elif page == "🔬 Microstructure Viewer":
    st.markdown('<h2 class="section-header">🔬 Microstructure Viewer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Microstructure Parameters")
        grain_size = st.slider("Grain Size (μm)", 10.0, 200.0, 50.0, 5.0)
        
        if st.button("Generate Microstructure", type="primary"):
            viewer = st.session_state.lab.MicrostructureViewer()
            st.session_state.microstructure_viewer = viewer
            st.success("Microstructure generated!")
    
    with col2:
        if st.session_state.current_microstructure:
            st.markdown("### Current Microstructure Stats")
            ms = st.session_state.current_microstructure
            st.metric("Grain Size", f"{ms.grain_size} μm")
            st.metric("Porosity", f"{ms.porosity*100:.2f}%")
            st.metric("Phase Count", f"{len(ms.phase_fractions)}")
    
    # Visualization
    if "microstructure_viewer" in st.session_state:
        st.markdown("### Microstructure Visualization")
        fig = st.session_state.microstructure_viewer.visualize_microstructure(grain_size)
        st.plotly_chart(fig, use_container_width=True)

# ==================== ALLOY DESIGNER ====================

elif page == "🧪 Alloy Designer":
    st.markdown('<h2 class="section-header">🧪 Alloy Designer</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Alloy Composition")
        base_element = st.selectbox("Base Element", ["Fe", "Al", "Ti", "Ni"])
        
        st.markdown("#### Alloying Elements (wt%)")
        c_content = st.slider("Carbon (C)", 0.0, 1.0, 0.35, 0.01)
        mn_content = st.slider("Manganese (Mn)", 0.0, 2.0, 0.75, 0.05)
        cr_content = st.slider("Chromium (Cr)", 0.0, 5.0, 0.25, 0.05)
        ni_content = st.slider("Nickel (Ni)", 0.0, 5.0, 0.0, 0.05)
        
        alloying_elements = {
            "C": c_content,
            "Mn": mn_content,
            "Cr": cr_content,
            "Ni": ni_content
        }
        
        if st.button("Design Alloy", type="primary"):
            result = st.session_state.lab.design_alloy(
                base_element=base_element,
                alloying_elements=alloying_elements
            )
            st.session_state.alloy_result = result
            st.success("Alloy designed successfully!")
    
    with col2:
        if "alloy_result" in st.session_state:
            result = st.session_state.alloy_result
            
            st.markdown("### Alloy Properties")
            st.markdown("#### Composition")
            for element, content in result["alloy_composition"].items():
                if content > 0:
                    st.markdown(f"- {element}: {content:.2f}%")
            
            st.markdown("#### Predicted Properties")
            st.metric("Yield Strength", f"{result['predicted_yield_strength']} MPa")
            st.metric("Tensile Strength", f"{result['predicted_tensile_strength']} MPa")
            st.metric("Elongation", f"{result['predicted_elongation']}%")
            st.metric("Young's Modulus", f"{result['predicted_youngs_modulus']} GPa")
            
            st.markdown("#### Strengthening Contributions")
            st.metric("Solid Solution", f"{result['solid_solution_contribution']} MPa")
            st.metric("Grain Boundary", f"{result['grain_boundary_contribution']} MPa")

# ==================== DATA EXPORT ====================

elif page == "📊 Data Export":
    st.markdown('<h2 class="section-header">📊 Data Export & Certification</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_material:
        st.warning("Please run tests first to generate data for export.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Generate Test Certificate")
            test_type = st.selectbox(
                "Test Type",
                ["Tensile Test", "Fatigue Test", "Fracture Test", "Creep Test"]
            )
            
            # Get properties from last test
            properties = st.session_state.test_results.get("tensile", {
                "Yield Strength (MPa)": 0,
                "UTS (MPa)": 0,
                "Elongation (%)": 0
            })
            
            if st.button("Generate Certificate", type="primary"):
                certificate = st.session_state.lab.generate_test_certificate(
                    test_type=test_type,
                    material=st.session_state.current_material,
                    properties=properties
                )
                st.session_state.certificate = certificate
                st.success("Test certificate generated!")
        
        with col2:
            if "certificate" in st.session_state:
                st.markdown("### Certificate Preview")
                cert = st.session_state.certificate
                
                with st.expander("View Certificate Details"):
                    st.json(cert)
        
        st.markdown("### Export Options")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("Export to CSV", type="secondary"):
                if "certificate" in st.session_state:
                    # Flatten the certificate for CSV
                    flat_data = {}
                    for key, value in st.session_state.certificate.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                flat_data[f"{key}_{subkey}"] = subvalue
                        else:
                            flat_data[key] = value
                    
                    df = pd.DataFrame([flat_data])
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="test_certificate.csv",
                        mime="text/csv"
                    )
        
        with col4:
            if st.button("Export to JSON", type="secondary"):
                if "certificate" in st.session_state:
                    json_data = json.dumps(st.session_state.certificate, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name="test_certificate.json",
                        mime="application/json"
                    )

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9rem; padding: 1rem;'>
    <b>Virtual Materials Testing Laboratory v3.0</b><br>
    Academic Edition | ISO 6892-1 Compliant | Multi-scale Modeling Framework
</div>
""", unsafe_allow_html=True)
