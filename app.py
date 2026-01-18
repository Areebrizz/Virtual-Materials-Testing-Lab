"""
VIRTUAL MATERIALS LAB - Advanced Materials Science Simulator
Version 3.0 | ISO 6892-1 Compliant | Multi-scale Modeling Framework
"""

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
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')

# ==================== CORE MATERIALS SCIENCE MODELS ====================

class CrystalStructure(Enum):
    """Crystallographic structure definitions"""
    BCC = "Body-Centered Cubic"
    FCC = "Face-Centered Cubic"
    HCP = "Hexagonal Close-Packed"
    COMPOSITE = "Composite/Amorphous"

class MaterialClass(Enum):
    """Material classification"""
    STEEL = "Carbon/Low-Alloy Steel"
    ALUMINUM = "Aluminum Alloy"
    TITANIUM = "Titanium Alloy"
    COMPOSITE = "Fiber-Reinforced Composite"
    SUPERALLOY = "Nickel-based Superalloy"

@dataclass
class Microstructure:
    """Advanced microstructure representation with crystallographic data"""
    grain_size: float  # μm
    phase_fractions: Dict[str, float]  # α/β, ferrite/austenite
    defect_density: float  # defects/mm²
    porosity: float  # volume fraction
    inclusion_size: float  # μm
    inclusion_volume_fraction: float
    crystal_structure: CrystalStructure
    texture_coefficient: float = 1.0  # anisotropy factor
    grain_size_distribution: str = "log-normal"
    twin_density: float = 0.0  # twins per grain
    
    def calculate_hall_peetch(self) -> float:
        """Hall-Petch strengthening coefficient"""
        if self.grain_size > 0:
            return 500 / np.sqrt(self.grain_size)  # MPa√mm
        return 0

@dataclass
class HeatTreatment:
    """Advanced heat treatment parameters with phase transformation kinetics"""
    quenching_rate: float  # °C/s
    tempering_temperature: float  # °C
    tempering_time: float  # hours
    austenitizing_temp: float = 950.0  # °C
    cooling_medium: str = "oil"  # oil, water, air
    precipitation_temp: float = 500.0  # °C
    aging_time: float = 8.0  # hours
    martensite_start: float = 300.0  # °C
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
    """Comprehensive material properties database"""
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
    
    # Advanced properties
    stacking_fault_energy: float = 50.0  # mJ/m²
    burgers_vector: float = 0.25  # nm
    shear_modulus: float = None
    lattice_parameter: float = None
    
    def __post_init__(self):
        if self.shear_modulus is None:
            self.shear_modulus = self.youngs_modulus / (2 * (1 + self.poissons_ratio))

class VirtualMaterialsLab:
    """Main laboratory class integrating all modules"""
    
    def __init__(self):
        self.materials_db = self._initialize_materials_database()
        self.current_material = None
        self.current_microstructure = None
        self.current_heat_treatment = None
        self.test_results = {}
        
    def _initialize_materials_database(self) -> Dict[str, MaterialProperties]:
        """Initialize with ASM Handbook data"""
        return {
            "AISI 1045": MaterialProperties(
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
            )
        }
    
    # ==================== SAMPLE PREPARATION STATION ====================
    
    def design_microstructure(self, material: str, 
                            grain_size: float = 50.0,
                            phase_fraction: Dict[str, float] = None,
                            porosity: float = 0.01,
                            inclusion_size: float = 10.0) -> Microstructure:
        """Advanced microstructure designer with phase transformation"""
        
        if phase_fraction is None:
            if material == "Ti-6Al-4V":
                phase_fraction = {"alpha": 0.9, "beta": 0.1}
            elif "steel" in material.lower():
                phase_fraction = {"ferrite": 0.85, "pearlite": 0.15}
            else:
                phase_fraction = {"matrix": 1.0}
        
        crystal_structure = {
            "AISI 1045": CrystalStructure.BCC,
            "Al 6061-T6": CrystalStructure.FCC,
            "Ti-6Al-4V": CrystalStructure.HCP
        }.get(material, CrystalStructure.BCC)
        
        self.current_microstructure = Microstructure(
            grain_size=grain_size,
            phase_fractions=phase_fraction,
            defect_density=1000.0,
            porosity=porosity,
            inclusion_size=inclusion_size,
            inclusion_volume_fraction=0.005,
            crystal_structure=crystal_structure,
            texture_coefficient=0.85
        )
        
        return self.current_microstructure
    
    def apply_heat_treatment(self, quenching_rate: float = 100.0,
                           tempering_temp: float = 600.0,
                           tempering_time: float = 2.0) -> HeatTreatment:
        """Advanced heat treatment simulator with TTT/CCT kinetics"""
        
        self.current_heat_treatment = HeatTreatment(
            quenching_rate=quenching_rate,
            tempering_temperature=tempering_temp,
            tempering_time=tempering_time,
            cooling_medium="oil" if quenching_rate < 50 else "water"
        )
        
        return self.current_heat_treatment
    
    # ==================== TENSILE TESTING MODULE ====================
    
    class TensileTester:
        """Advanced tensile testing with true stress-strain and necking simulation"""
        
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
            """Generate stress-strain curve with advanced constitutive models"""
            
            # Base elastic-plastic response
            eps = np.linspace(0, 0.25, 1000)
            
            if constitutive_model == "hollomon":
                # Hollomon power law: σ = Kεⁿ
                K = self.material.tensile_strength * (np.exp(self.material.elongation/100))**0.2
                n = 0.1 + 0.2 * (self.material.elongation/100)
                
                # Elastic region
                elastic_limit = self.material.yield_strength / self.material.youngs_modulus
                mask_elastic = eps <= elastic_limit
                mask_plastic = eps > elastic_limit
                
                stress = np.zeros_like(eps)
                stress[mask_elastic] = self.material.youngs_modulus * 1000 * eps[mask_elastic]
                stress[mask_plastic] = K * (eps[mask_plastic] - elastic_limit)**n
                
            elif constitutive_model == "voce":
                # Voce saturation hardening: σ = σ₀ + Q(1 - exp(-bε))
                sigma_0 = self.material.yield_strength
                Q = self.material.tensile_strength - sigma_0
                b = 20.0
                
                elastic_limit = sigma_0 / self.material.youngs_modulus
                mask_elastic = eps <= elastic_limit
                mask_plastic = eps > elastic_limit
                
                stress = np.zeros_like(eps)
                stress[mask_elastic] = self.material.youngs_modulus * 1000 * eps[mask_elastic]
                stress[mask_plastic] = (sigma_0 + Q * (1 - np.exp(-b * (eps[mask_plastic] - elastic_limit))))
            
            # Apply Hall-Petch strengthening
            if self.microstructure:
                hp_strength = self.microstructure.calculate_hall_peetch()
                stress += hp_strength * (eps > elastic_limit)
            
            # Temperature effect (Arrhenius-type)
            if temperature > 20:
                Q_activation = 300e3  # J/mol
                R = 8.314
                temp_factor = np.exp(-Q_activation/R * (1/(temperature+273) - 1/293))
                stress *= temp_factor
            
            # Strain rate effect (Johnson-Cook)
            eps0_dot = 0.001
            C = 0.014  # strain rate sensitivity
            strain_rate_factor = (1 + C * np.log(strain_rate/eps0_dot))
            stress *= strain_rate_factor
            
            self.engineering_stress_strain = (eps, stress)
            
            # Convert to true stress-strain
            true_strain = np.log(1 + eps)
            true_stress = stress * (1 + eps)
            
            self.true_stress_strain = (true_strain, true_stress)
            
            # Find necking point (Considère criterion: dσ/dε = σ)
            plastic_range = eps > elastic_limit
            if np.any(plastic_range):
                eps_plastic = eps[plastic_range]
                stress_plastic = stress[plastic_range]
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
            
            # Young's modulus from initial slope
            initial_slope = np.polyfit(eps[:50], stress[:50], 1)[0]
            E = initial_slope / 1000  # Convert to GPa
            
            # 0.2% offset yield strength
            offset_strain = eps + 0.002
            offset_line = E * 1000 * (eps - 0.002)
            intersect_idx = np.where(stress >= offset_line)[0]
            sigma_y = stress[intersect_idx[0]] if len(intersect_idx) > 0 else stress[0]
            
            # Ultimate tensile strength
            sigma_uts = np.max(stress)
            uts_idx = np.argmax(stress)
            
            # Uniform and total elongation
            uniform_elongation = eps[uts_idx]
            fracture_strain = eps[-1]
            
            # Strain hardening exponent (n-value)
            plastic_strain = eps[eps > sigma_y/(E*1000)] - sigma_y/(E*1000)
            plastic_stress = stress[eps > sigma_y/(E*1000)]
            if len(plastic_strain) > 10:
                log_eps = np.log(plastic_strain)
                log_sigma = np.log(plastic_stress)
                n_value = np.polyfit(log_eps, log_sigma, 1)[0]
            else:
                n_value = 0.1
            
            return {
                "Young's Modulus (GPa)": E,
                "Yield Strength (MPa)": sigma_y,
                "UTS (MPa)": sigma_uts,
                "Uniform Elongation (%)": uniform_elongation * 100,
                "Total Elongation (%)": fracture_strain * 100,
                "Strain Hardening Exponent (n)": n_value,
                "Necking Strain (%)": self.necking_point[0]*100 if self.necking_point else 0
            }
        
        def visualize_curve(self, show_true: bool = True):
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
            if show_true:
                fig.add_trace(
                    go.Scatter(x=true_strain*100, y=true_stress, mode='lines',
                              name='True', line=dict(color='red', width=2)),
                    row=1, col=2
                )
            
            # Strain hardening rate
            plastic_range = eps > (self.material.yield_strength/(self.material.youngs_modulus*1000))
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
                height=800,
                showlegend=True,
                title_text=f"Tensile Test Results - {self.material.__class__.__name__}"
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
        """Advanced fatigue life prediction with crack growth simulation"""
        
        def __init__(self, material_props: MaterialProperties,
                    microstructure: Microstructure = None):
            self.material = material_props
            self.microstructure = microstructure
            self.SN_data = None
            self.crack_growth_data = None
            
        def generate_SN_curve(self, R_ratio: float = -1.0,
                            surface_finish: str = "polished",
                            environment: str = "air",
                            reliability: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
            """Generate S-N curve with statistical reliability"""
            
            # Basquin equation: σ_a = σ_f' * (2N_f)^b
            sigma_f_prime = self.material.tensile_strength * 1.5  # Approx
            b = -0.085  # Basquin exponent for steels
            
            N_cycles = np.logspace(3, 8, 50)  # 1e3 to 1e8 cycles
            stress_amp = sigma_f_prime * (2 * N_cycles) ** b
            
            # Mean stress effect (Goodman correction)
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
            
            # Reliability factor (statistical scatter)
            if reliability != 0.5:
                z = stats.norm.ppf(reliability)
                scatter_band = 1.2  # Typical for metals
                stress_amp *= (1 - z * 0.05 * scatter_band)
            
            # Fatigue limit
            fatigue_limit = self.material.fatigue_limit
            stress_amp[stress_amp < fatigue_limit] = fatigue_limit
            
            self.SN_data = (N_cycles, stress_amp)
            return N_cycles, stress_amp
        
        def paris_law_crack_growth(self, initial_crack: float = 0.1,
                                 final_crack: float = 10.0,
                                 delta_K_th: float = 5.0,
                                 K_c: float = None) -> Tuple[np.ndarray, np.ndarray]:
            """Paris-Erdogan crack growth simulation"""
            
            if K_c is None:
                K_c = self.material.fracture_toughness
            
            # Paris law constants (typical for steels)
            C = 6.9e-12  # mm/cycle/(MPa√m)^m
            m = 3.0
            
            # Crack lengths
            a = np.linspace(initial_crack, final_crack, 1000)
            
            # Stress intensity factor range
            # Simplified: ΔK = Δσ * √(πa)
            delta_sigma = 200  # MPa, typical fatigue loading
            delta_K = delta_sigma * np.sqrt(np.pi * a)
            
            # Apply threshold and fracture toughness limits
            valid = (delta_K > delta_K_th) & (delta_K < K_c)
            da_dN = np.zeros_like(a)
            da_dN[valid] = C * (delta_K[valid] ** m)
            
            # Calculate number of cycles
            N_cycles = np.zeros_like(a)
            if np.any(valid):
                integrand = 1 / (C * (delta_sigma * np.sqrt(np.pi * a[valid]) ** m))
                N_integrated = integrate.cumtrapz(integrand, a[valid], initial=0)
                N_cycles[valid] = N_integrated
            
            self.crack_growth_data = (a, da_dN, N_cycles)
            return a, da_dN, N_cycles
        
        def fracture_surface_simulation(self, crack_length: float = 5.0) -> go.Figure:
            """Generate synthetic fracture surface visualization"""
            
            # Create synthetic fracture surface with features
            x = np.linspace(-10, 10, 400)
            y = np.linspace(-10, 10, 400)
            X, Y = np.meshgrid(x, y)
            
            # Beach marks (fatigue striations)
            Z = np.zeros_like(X)
            
            # Add radial beach marks from crack origin
            r = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            
            # Fatigue striations
            for i in range(1, 20):
                Z += 0.2 * np.sin(2 * np.pi * r / (1 + i*0.5) + theta)
            
            # Overload marks
            Z += 0.5 * np.exp(-(r**2)/50) * np.sin(5*theta)
            
            # Final fracture region
            final_frac = r > crack_length
            Z[final_frac] += 3.0 * np.random.randn(*Z[final_frac].shape) * 0.1
            
            fig = go.Figure(data=[
                go.Surface(z=Z, x=X, y=Y, 
                          colorscale='Viridis',
                          contours={
                              "z": {"show": True, "usecolormap": True}
                          })
            ])
            
            fig.update_layout(
                title="Fracture Surface SEM Simulation",
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)",
                    zaxis_title="Height (μm)",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1)
                    )
                ),
                height=600
            )
            
            return fig
    
    # ==================== FRACTURE TOUGHNESS MODULE ====================
    
    class FractureToughnessTester:
        """Advanced fracture mechanics with plastic zone simulation"""
        
        def __init__(self, material_props: MaterialProperties):
            self.material = material_props
            
        def calculate_stress_field(self, K_I: float = 30.0,
                                 distance: float = 10.0,
                                 theta: np.ndarray = None) -> Dict[str, np.ndarray]:
            """Calculate crack tip stress field (Mode I)"""
            
            if theta is None:
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
        
        def estimate_plastic_zone(self, K_I: float = 30.0,
                                plane_stress: bool = True) -> float:
            """Estimate plastic zone size"""
            
            sigma_y = self.material.yield_strength
            
            if plane_stress:
                r_p = (1/(2*np.pi)) * (K_I/sigma_y)**2
            else:
                r_p = (1/(6*np.pi)) * (K_I/sigma_y)**2
            
            return r_p
        
        def visualize_crack_tip(self, K_I: float = 30.0):
            """Interactive crack tip stress field visualization"""
            
            stress_field = self.calculate_stress_field(K_I)
            r_p = self.estimate_plastic_zone(K_I)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("σ_xx Stress Field",
                              "σ_yy Stress Field",
                              "Von Mises Stress",
                              "Plastic Zone"),
                specs=[[{'type': 'surface'}, {'type': 'surface'}],
                       [{'type': 'surface'}, {'type': 'scatter'}]]
            )
            
            # Convert polar to Cartesian for plotting
            x = stress_field['r'] * np.cos(stress_field['theta'])
            y = stress_field['r'] * np.sin(stress_field['theta'])
            
            # σ_xx
            fig.add_trace(
                go.Surface(z=stress_field['sigma_xx'], x=x, y=y,
                          colorscale='RdBu', showscale=False),
                row=1, col=1
            )
            
            # σ_yy
            fig.add_trace(
                go.Surface(z=stress_field['sigma_yy'], x=x, y=y,
                          colorscale='RdBu', showscale=False),
                row=1, col=2
            )
            
            # von Mises
            fig.add_trace(
                go.Surface(z=stress_field['sigma_vm'], x=x, y=y,
                          colorscale='Viridis', showscale=False),
                row=2, col=1
            )
            
            # Plastic zone
            theta_circle = np.linspace(0, 2*np.pi, 100)
            x_circle = r_p * np.cos(theta_circle)
            y_circle = r_p * np.sin(theta_circle)
            
            fig.add_trace(
                go.Scatter(x=x_circle, y=y_circle,
                          mode='lines', fill='toself',
                          fillcolor='rgba(255,0,0,0.2)',
                          line=dict(color='red'),
                          name='Plastic Zone'),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text=f"Crack Tip Stress Fields - K_I = {K_I} MPa√m",
                showlegend=True
            )
            
            return fig
    
    # ==================== CREEP TESTING MODULE ====================
    
    class CreepTester:
        """Advanced creep deformation and rupture prediction"""
        
        def __init__(self, material_props: MaterialProperties):
            self.material = material_props
            
        def creep_deformation(self, stress: float = 100.0,
                            temperature: float = 600.0,
                            time_hours: float = 1000.0) -> np.ndarray:
            """Calculate creep strain using Norton's law"""
            
            # Norton's law: ε_creep = A σ^n t
            # A = A0 exp(-Q/RT)
            
            # Material-dependent parameters
            if self.material.crystal_structure == CrystalStructure.FCC:
                n = 5.0
                Q = 250e3  # J/mol
                A0 = 1e-10
            else:
                n = 4.0
                Q = 300e3
                A0 = 1e-11
            
            R = 8.314  # J/mol·K
            T = temperature + 273.15  # K
            
            # Time in seconds
            t_seconds = time_hours * 3600
            
            # Time array
            t = np.linspace(0, t_seconds, 1000)
            
            # Creep strain
            A = A0 * np.exp(-Q/(R*T))
            epsilon_creep = A * (stress ** n) * t
            
            return t/3600, epsilon_creep * 100  # Return hours and % strain
        
        def larson_miller_parameter(self, stress: np.ndarray = None,
                                  T: float = 600.0,
                                  t_r: float = 1000.0) -> float:
            """Calculate Larson-Miller parameter"""
            
            if stress is None:
                stress = np.linspace(50, 300, 50)
            
            # LMP = T(C + log t_r)
            C = 20  # Typical for steels
            LMP = (T + 273.15) * (C + np.log10(t_r))
            
            return stress, LMP
        
        def stress_rupture_curve(self, temperature: float = 600.0):
            """Generate stress-rupture curves"""
            
            times = np.array([10, 100, 1000, 10000, 100000])  # hours
            stresses = np.linspace(50, 300, 50)
            
            fig = go.Figure()
            
            for t_r in times:
                # Simplified model: σ = a - b log(t_r)
                sigma_max = self.material.tensile_strength * np.exp(-0.001 * temperature)
                b = 0.1 * sigma_max
                
                sigma_rupture = sigma_max - b * np.log10(t_r)
                
                fig.add_trace(
                    go.Scatter(x=[np.log10(t_r)], y=[sigma_rupture],
                              mode='markers+text',
                              name=f'{t_r} hours',
                              text=[f'{t_r}h'],
                              textposition='top center',
                              marker=dict(size=10))
                )
            
            # Add iso-LMP lines
            for LMP in [18000, 20000, 22000]:
                sigma = self.material.tensile_strength * np.exp(-0.0005 * LMP/100)
                fig.add_trace(
                    go.Scatter(x=[1, 5], y=[sigma, sigma],
                              mode='lines',
                              line=dict(dash='dash', width=1),
                              name=f'LMP={LMP}',
                              showlegend=True)
                )
            
            fig.update_layout(
                title=f"Stress-Rupture Curves at {temperature}°C",
                xaxis_title="Log Time (hours)",
                yaxis_title="Stress (MPa)",
                height=500
            )
            
            return fig
    
    # ==================== MICROSTRUCTURE VIEWER ====================
    
    class MicrostructureViewer:
        """Advanced microstructure generation and visualization"""
        
        def __init__(self):
            self.grains = None
            self.phases = None
            
        def generate_voronoi_microstructure(self, grain_size: float = 50.0,
                                          phase_fractions: Dict[str, float] = None,
                                          size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
            """Generate synthetic microstructure using Voronoi tessellation"""
            
            if phase_fractions is None:
                phase_fractions = {'alpha': 0.9, 'beta': 0.1}
            
            # Number of grains
            n_grains = int((size * size) / (grain_size ** 2))
            
            # Random grain centers
            points = np.random.rand(n_grains, 2) * size
            
            # Create grid
            x, y = np.meshgrid(np.arange(size), np.arange(size))
            grid_points = np.column_stack([x.ravel(), y.ravel()])
            
            # Assign each point to nearest grain center
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            distances, grain_indices = tree.query(grid_points)
            
            # Reshape to image
            grain_map = grain_indices.reshape((size, size))
            
            # Assign phases
            n_alpha = int(n_grains * phase_fractions.get('alpha', 0.9))
            phase_map = np.zeros_like(grain_map)
            
            for i in range(n_grains):
                mask = grain_map == i
                if i < n_alpha:
                    phase_map[mask] = 1  # alpha phase
                else:
                    phase_map[mask] = 2  # beta phase
            
            # Add grain boundaries
            from scipy.ndimage import sobel
            gb_x = sobel(grain_map, axis=0)
            gb_y = sobel(grain_map, axis=1)
            grain_boundaries = np.sqrt(gb_x**2 + gb_y**2) > 0
            
            return grain_map, phase_map, grain_boundaries
        
        def visualize_microstructure_3d(self, grain_size: float = 50.0):
            """3D visualization of microstructure with phases"""
            
            grain_map, phase_map, boundaries = self.generate_voronoi_microstructure(grain_size)
            
            # Create 3D surface with height representing phases
            x, y = np.mgrid[0:grain_map.shape[0], 0:grain_map.shape[1]]
            
            fig = go.Figure(data=[
                go.Surface(z=phase_map + 0.1*boundaries,
                          x=x, y=y,
                          colorscale=[[0, 'lightblue'], [0.5, 'blue'], [1, 'darkblue']],
                          opacity=0.8,
                          contours={
                              "z": {"show": True, "usecolormap": True}
                          })
            ])
            
            fig.update_layout(
                title=f"3D Microstructure Visualization - Grain Size: {grain_size}μm",
                scene=dict(
                    xaxis_title="X (μm)",
                    yaxis_title="Y (μm)",
                    zaxis_title="Phase ID",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    )
                ),
                height=600
            )
            
            return fig
        
        def ebsd_simulation(self, grain_size: float = 50.0):
            """Generate synthetic EBSD-like patterns"""
            
            size = 400
            grain_map, _, _ = self.generate_voronoi_microstructure(grain_size, size=size)
            
            # Create orientation map (simplified)
            orientation_map = np.zeros((size, size, 3))
            
            unique_grains = np.unique(grain_map)
            for grain in unique_grains:
                mask = grain_map == grain
                # Random orientation (Euler angles)
                phi1, Phi, phi2 = np.random.rand(3) * 360
                orientation_map[mask, 0] = phi1
                orientation_map[mask, 1] = Phi
                orientation_map[mask, 2] = phi2
            
            # Create IPF coloring (simplified)
            ipf_z = np.zeros((size, size, 3))
            for i in range(3):
                ipf_z[:,:,i] = orientation_map[:,:,i] / 360
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=("IPF-Z Map", "Grain Boundaries", "Misorientation Map")
            )
            
            # IPF Map
            fig.add_trace(
                go.Heatmap(z=ipf_z[:,:,0], colorscale='Viridis'),
                row=1, col=1
            )
            
            # Grain boundaries
            from scipy.ndimage import sobel
            gb = np.sqrt(sobel(grain_map, axis=0)**2 + sobel(grain_map, axis=1)**2) > 0
            fig.add_trace(
                go.Heatmap(z=gb, colorscale='gray'),
                row=1, col=2
            )
            
            # Misorientation
            misorientation = np.abs(np.gradient(grain_map)[0])
            fig.add_trace(
                go.Heatmap(z=misorientation, colorscale='hot'),
                row=1, col=3
            )
            
            fig.update_layout(
                height=400,
                title_text="Synthetic EBSD Analysis"
            )
            
            return fig
    
    # ==================== ALLOY DESIGNER ====================
    
    def design_alloy(self, base_element: str = "Fe",
                    alloying_elements: Dict[str, float] = None,
                    target_properties: Dict[str, float] = None) -> Dict[str, Any]:
        """Advanced alloy design using empirical models"""
        
        if alloying_elements is None:
            alloying_elements = {"C": 0.45, "Mn": 0.75, "Si": 0.25, "Cr": 0.25}
        
        if target_properties is None:
            target_properties = {"yield_strength": 500, "elongation": 15}
        
        # Empirical strengthening models
        base_strength = 200  # MPa for pure Fe
        
        # Solid solution strengthening
        ss_coefficients = {
            "C": 5000, "Mn": 80, "Si": 60, "Cr": 50,
            "Ni": 40, "Mo": 100, "V": 300, "Ti": 400
        }
        
        ss_strength = 0
        for element, wt_pct in alloying_elements.items():
            if element in ss_coefficients:
                ss_strength += ss_coefficients[element] * wt_pct
        
        # Precipitation strengthening (simplified)
        precip_strength = 0
        if "V" in alloying_elements or "Ti" in alloying_elements:
            precip_strength = 200 * sum(alloying_elements.get(e, 0) for e in ["V", "Ti", "Nb"])
        
        # Grain boundary strengthening (Hall-Petch)
        grain_size = 20  # μm
        gb_strength = 500 / np.sqrt(grain_size)
        
        # Total yield strength
        predicted_yield = base_strength + ss_strength + precip_strength + gb_strength
        
        # Estimate elongation
        predicted_elongation = 30 - 20 * (predicted_yield / 1000)
        
        # Estimate other properties
        predicted_E = 200 + 10 * alloying_elements.get("Cr", 0)  # GPa
        
        return {
            "alloy_composition": alloying_elements,
            "predicted_yield_strength": predicted_yield,
            "predicted_tensile_strength": predicted_yield * 1.2,
            "predicted_elongation": max(5, predicted_elongation),
            "predicted_youngs_modulus": predicted_E,
            "solid_solution_contribution": ss_strength,
            "precipitation_contribution": precip_strength,
            "grain_boundary_contribution": gb_strength
        }
    
    # ==================== DATA EXPORT & CERTIFICATION ====================
    
    def generate_test_certificate(self, test_type: str,
                                material: str,
                                properties: Dict[str, float],
                                iso_standard: str = "ISO 6892-1") -> Dict[str, Any]:
        """Generate ISO-compliant test certificate"""
        
        certificate = {
            "test_laboratory": "Virtual Materials Testing Lab v3.0",
            "iso_standard": iso_standard,
            "test_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "material_identification": material,
            "test_type": test_type,
            "test_conditions": {
                "temperature": "23 ± 2°C",
                "humidity": "50 ± 10%",
                "strain_rate": "0.001 s⁻¹"
            },
            "mechanical_properties": properties,
            "measurement_uncertainty": {
                "yield_strength": "± 2%",
                "tensile_strength": "± 1%",
                "elongation": "± 5%",
                "youngs_modulus": "± 3%"
            },
            "calibration": {
                "force_cell": "ISO 7500-1 compliant",
                "extensometer": "ISO 9513 compliant",
                "last_calibration": pd.Timestamp.now() - pd.Timedelta(days=30)
            },
            "signature": {
                "test_engineer": "Virtual Materials Scientist",
                "approval": "ISO/IEC 17025 compliant"
            }
        }
        
        return certificate
    
    def export_to_csv(self, data: Dict[str, Any], filename: str = "test_results.csv"):
        """Export test results to CSV format"""
        
        df = pd.DataFrame([data])
        df.to_csv(filename, index=False)
        return f"Data exported to {filename}"
    
    def export_to_json(self, data: Dict[str, Any], filename: str = "test_results.json"):
        """Export test results to JSON format"""
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return f"Data exported to {filename}"

# ==================== MAIN APPLICATION ====================

def main():
    """Demonstrate complete lab functionality"""
    
    print("=" * 70)
    print("VIRTUAL MATERIALS TESTING LABORATORY v3.0")
    print("Multi-scale Materials Science Simulation Platform")
    print("=" * 70)
    
    # Initialize lab
    lab = VirtualMaterialsLab()
    
    # 1. SAMPLE PREPARATION
    print("\n1. SAMPLE PREPARATION STATION")
    print("-" * 40)
    
    # Select material
    material_name = "AISI 1045"
    material = lab.materials_db[material_name]
    print(f"Selected material: {material_name}")
    
    # Design microstructure
    microstructure = lab.design_microstructure(
        material_name,
        grain_size=25.0,
        phase_fraction={"ferrite": 0.85, "pearlite": 0.15},
        porosity=0.005,
        inclusion_size=15.0
    )
    print(f"Designed microstructure: Grain size = {microstructure.grain_size}μm")
    print(f"Hall-Petch strengthening: {microstructure.calculate_hall_peetch():.1f} MPa")
    
    # Apply heat treatment
    heat_treatment = lab.apply_heat_treatment(
        quenching_rate=80.0,
        tempering_temp=550.0,
        tempering_time=2.0
    )
    print(f"Applied heat treatment: {heat_treatment.cooling_medium} quench")
    
    # 2. TENSILE TESTING
    print("\n2. TENSILE TESTING MODULE")
    print("-" * 40)
    
    tensile_tester = lab.TensileTester(material, microstructure)
    eps, stress = tensile_tester.generate_stress_strain_curve(
        constitutive_model="voce",
        temperature=20.0,
        strain_rate=0.001
    )
    
    properties = tensile_tester.calculate_mechanical_properties()
    print("\nMechanical Properties:")
    for prop, value in properties.items():
        print(f"  {prop}: {value:.2f}")
    
    # Generate visualization
    fig_tensile = tensile_tester.visualize_curve(show_true=True)
    fig_tensile.write_html("tensile_test_results.html")
    print("Tensile test visualization saved to tensile_test_results.html")
    
    # 3. FATIGUE TESTING
    print("\n3. FATIGUE TESTING MODULE")
    print("-" * 40)
    
    fatigue_tester = lab.FatigueTester(material, microstructure)
    N_cycles, stress_amp = fatigue_tester.generate_SN_curve(
        R_ratio=-1.0,
        surface_finish="polished",
        reliability=0.95
    )
    
    print(f"Fatigue limit: {material.fatigue_limit:.1f} MPa")
    print(f"10^6 cycle strength: {stress_amp[-1]:.1f} MPa")
    
    # Crack growth simulation
    a, da_dN, N = fatigue_tester.paris_law_crack_growth(
        initial_crack=0.1,
        final_crack=8.0
    )
    print(f"Crack growth from 0.1mm to 8mm: {N[-1]:,.0f} cycles")
    
    # Fracture surface simulation
    fig_fracture = fatigue_tester.fracture_surface_simulation(crack_length=5.0)
    fig_fracture.write_html("fracture_surface.html")
    
    # 4. FRACTURE TOUGHNESS
    print("\n4. FRACTURE TOUGHNESS TESTING")
    print("-" * 40)
    
    fracture_tester = lab.FractureToughnessTester(material)
    r_p = fracture_tester.estimate_plastic_zone(K_I=40.0)
    print(f"Plastic zone size for K_I=40 MPa√m: {r_p:.3f} mm")
    
    fig_crack = fracture_tester.visualize_crack_tip(K_I=40.0)
    fig_crack.write_html("crack_tip_stress_field.html")
    
    # 5. CREEP TESTING
    print("\n5. CREEP TESTING MODULE")
    print("-" * 40)
    
    creep_tester = lab.CreepTester(material)
    t, eps_creep = creep_tester.creep_deformation(
        stress=150.0,
        temperature=500.0,
        time_hours=10000.0
    )
    print(f"Creep strain after 10,000h at 150MPa, 500°C: {eps_creep[-1]:.3f}%")
    
    fig_creep = creep_tester.stress_rupture_curve(temperature=500.0)
    fig_creep.write_html("stress_rupture_curves.html")
    
    # 6. MICROSTRUCTURE ANALYSIS
    print("\n6. MICROSTRUCTURE VIEWER")
    print("-" * 40)
    
    microstructure_viewer = lab.MicrostructureViewer()
    fig_micro = microstructure_viewer.visualize_microstructure_3d(grain_size=30.0)
    fig_micro.write_html("microstructure_3d.html")
    
    fig_ebsd = microstructure_viewer.ebsd_simulation(grain_size=30.0)
    fig_ebsd.write_html("ebsd_simulation.html")
    
    # 7. ALLOY DESIGN
    print("\n7. ALLOY DESIGNER")
    print("-" * 40)
    
    new_alloy = lab.design_alloy(
        base_element="Fe",
        alloying_elements={"C": 0.35, "Mn": 1.0, "Si": 0.3, "Cr": 1.5, "Mo": 0.25},
        target_properties={"yield_strength": 800, "elongation": 12}
    )
    
    print("\nDesigned Alloy Properties:")
    for key, value in new_alloy.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # 8. GENERATE TEST CERTIFICATE
    print("\n8. TEST CERTIFICATION")
    print("-" * 40)
    
    certificate = lab.generate_test_certificate(
        test_type="Tensile Test",
        material=material_name,
        properties=properties
    )
    
    lab.export_to_json(certificate, "test_certificate.json")
    print("ISO-compliant test certificate generated: test_certificate.json")
    
    # Export test data
    test_data = {
        "material": material_name,
        "microstructure": {
            "grain_size": microstructure.grain_size,
            "phase_fractions": microstructure.phase_fractions
        },
        "mechanical_properties": properties,
        "fatigue_properties": {
            "fatigue_limit": material.fatigue_limit,
            "SN_curve": list(zip(N_cycles, stress_amp))
        }
    }
    
    lab.export_to_csv(test_data, "complete_test_data.csv")
    print("Complete test data exported: complete_test_data.csv")
    
    print("\n" + "=" * 70)
    print("VIRTUAL MATERIALS LAB TESTING COMPLETE")
    print("All simulations executed successfully")
    print("=" * 70)
    
    return lab

if __name__ == "__main__":
    # Run the complete virtual materials lab
    lab_instance = main()
    
    print("\nTo visualize results, open the generated HTML files:")
    print("1. tensile_test_results.html - Interactive stress-strain curves")
    print("2. fracture_surface.html - 3D fracture surface simulation")
    print("3. crack_tip_stress_field.html - Crack tip stress fields")
    print("4. microstructure_3d.html - 3D microstructure visualization")
    print("5. ebsd_simulation.html - Synthetic EBSD patterns")
    print("6. stress_rupture_curves.html - Creep rupture predictions")
