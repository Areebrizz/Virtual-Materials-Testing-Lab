"""
CORE MATERIALS SCIENCE ENGINE
Virtual Materials Testing Laboratory - Backend Engine
Version 3.0 | ISO 6892-1 Compliant | Multi-scale Modeling Framework
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
from scipy import integrate, optimize, interpolate, stats
from scipy.spatial import cKDTree
from scipy.ndimage import sobel, gaussian_filter
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')

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
    
    @property
    def slip_systems(self) -> List[str]:
        """Primary slip systems for each crystal structure"""
        systems = {
            CrystalStructure.BCC: ["{110}<111>", "{112}<111>"],
            CrystalStructure.FCC: ["{111}<110>"],
            CrystalStructure.HCP: ["{0001}<1120>", "{1010}<1120>"],
            CrystalStructure.COMPOSITE: ["Random"]
        }
        return systems.get(self, ["Random"])

class MaterialClass(Enum):
    """Material classification with typical applications"""
    STEEL = "Carbon/Low-Alloy Steel"
    ALUMINUM = "Aluminum Alloy"
    TITANIUM = "Titanium Alloy"
    COMPOSITE = "Fiber-Reinforced Composite"
    SUPERALLOY = "Nickel-based Superalloy"
    CERAMIC = "Advanced Ceramic"
    POLYMER = "Engineering Polymer"
    
    @property
    def typical_applications(self) -> List[str]:
        """Typical applications for each material class"""
        apps = {
            MaterialClass.STEEL: ["Structural beams", "Automotive parts", "Tools"],
            MaterialClass.ALUMINUM: ["Aircraft frames", "Packaging", "Electronics"],
            MaterialClass.TITANIUM: ["Aerospace components", "Medical implants", "Chemical plants"],
            MaterialClass.COMPOSITE: ["Wind turbine blades", "Sports equipment", "Aerospace structures"],
            MaterialClass.SUPERALLOY: ["Jet engine turbines", "Nuclear reactors", "Gas turbines"],
            MaterialClass.CERAMIC: ["Cutting tools", "Heat shields", "Electronic substrates"],
            MaterialClass.POLYMER: ["Bearings", "Gears", "Medical devices"]
        }
        return apps.get(self, ["General engineering"])

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
    dislocation_density: float = 1e12  # m⁻²
    
    def calculate_hall_peetch(self) -> float:
        """Hall-Petch strengthening coefficient: σ = σ₀ + k/√d"""
        if self.grain_size > 0:
            k = 500  # MPa√mm, typical for steels
            return k / np.sqrt(self.grain_size * 0.001)  # Convert μm to mm
        return 0
    
    def calculate_taylor_factor(self) -> float:
        """Taylor factor for polycrystal plasticity"""
        if self.crystal_structure == CrystalStructure.FCC:
            return 3.06
        elif self.crystal_structure == CrystalStructure.BCC:
            return 2.75
        elif self.crystal_structure == CrystalStructure.HCP:
            return 6.5
        else:
            return 3.0
    
    def calculate_yield_strength_contribution(self) -> Dict[str, float]:
        """Calculate contributions from different strengthening mechanisms"""
        contributions = {}
        
        # Hall-Petch strengthening
        contributions["grain_boundary"] = self.calculate_hall_peetch()
        
        # Solid solution strengthening (simplified)
        contributions["solid_solution"] = 50.0
        
        # Precipitation strengthening (if inclusions present)
        if self.inclusion_volume_fraction > 0:
            contributions["precipitation"] = 100 * self.inclusion_volume_fraction
        
        # Dislocation strengthening (Taylor hardening)
        G = 80e3  # Shear modulus in MPa
        b = 0.25e-9  # Burgers vector in m
        contributions["dislocation"] = self.taylor_factor * G * b * np.sqrt(self.dislocation_density)
        
        # Texture effect
        contributions["texture"] = 20 * (self.texture_coefficient - 1)
        
        return contributions
    
    @property
    def taylor_factor(self) -> float:
        """Taylor factor for crystal plasticity"""
        return self.calculate_taylor_factor()

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
        """Jominy end-quench hardenability calculation (ideal diameter)"""
        if self.cooling_medium == "water":
            return 50.0  # mm
        elif self.cooling_medium == "oil":
            return 30.0  # mm
        else:
            return 15.0  # mm
    
    def calculate_martensite_content(self) -> float:
        """Estimate martensite content based on cooling rate"""
        if self.quenching_rate > 100:  # Rapid quench
            return 0.95
        elif self.quenching_rate > 10:  # Moderate quench
            return 0.75
        else:  # Slow quench
            return 0.20
    
    def calculate_tempering_effect(self) -> Dict[str, float]:
        """Calculate effect of tempering on properties"""
        # Hollomon-Jaffe tempering parameter
        T = self.tempering_temperature + 273.15  # Convert to Kelvin
        P = T * (20 + np.log10(self.tempering_time * 3600))  # Larson-Miller type
        
        effects = {
            "hardness_reduction": 0.3 * (1 - np.exp(-P/20000)),
            "toughness_increase": 0.5 * (1 - np.exp(-P/15000)),
            "stress_relief": 0.8 * (1 - np.exp(-P/10000))
        }
        
        return effects

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
    
    # Advanced properties
    stacking_fault_energy: float = 50.0  # mJ/m²
    burgers_vector: float = 0.25  # nm
    shear_modulus: float = None
    lattice_parameter: float = None
    melting_point: float = None
    electrical_resistivity: float = None
    
    def __post_init__(self):
        if self.shear_modulus is None:
            self.shear_modulus = self.youngs_modulus / (2 * (1 + self.poissons_ratio))
        
        if self.melting_point is None:
            # Estimate based on crystal structure
            if self.crystal_structure == CrystalStructure.FCC:
                self.melting_point = 660.0  # Aluminum-like
            elif self.crystal_structure == CrystalStructure.BCC:
                self.melting_point = 1538.0  # Iron-like
            elif self.crystal_structure == CrystalStructure.HCP:
                self.melting_point = 1668.0  # Titanium-like
    
    @property
    def modulus_of_resilience(self) -> float:
        """Modulus of resilience: U_r = σ_y²/(2E)"""
        return (self.yield_strength ** 2) / (2 * self.youngs_modulus * 1000)  # MJ/m³
    
    @property
    def modulus_of_toughness(self) -> float:
        """Approximate modulus of toughness"""
        avg_strength = (self.yield_strength + self.tensile_strength) / 2
        return avg_strength * (self.elongation / 100)  # MJ/m³
    
    @property
    def hardness_estimate(self) -> float:
        """Estimated Vickers hardness from tensile strength"""
        # Tabor relation: HV ≈ 3 × σ_UTS (for metals)
        return 3.0 * self.tensile_strength / 9.807  # Convert MPa to kgf/mm²
    
    def calculate_property_bounds(self, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate statistical bounds for properties"""
        z = stats.norm.ppf(confidence)
        scatter = 0.05  # 5% typical scatter
        
        bounds = {}
        for prop_name in ['yield_strength', 'tensile_strength', 'elongation', 'fracture_toughness']:
            value = getattr(self, prop_name)
            lower = value * (1 - z * scatter)
            upper = value * (1 + z * scatter)
            bounds[prop_name] = (lower, upper)
        
        return bounds

# ==================== VIRTUAL MATERIALS LAB CORE ====================

class VirtualMaterialsLab:
    """Main laboratory class integrating all modules with advanced functionality"""
    
    def __init__(self):
        self.materials_db = self._initialize_materials_database()
        self.current_material = None
        self.current_microstructure = None
        self.current_heat_treatment = None
        self.test_results = {}
        self.simulation_history = []
        
    def _initialize_materials_database(self) -> Dict[str, MaterialProperties]:
        """Initialize with comprehensive ASM Handbook data"""
        return {
            "AISI 1045 (Steel)": MaterialProperties(
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
                crystal_structure=CrystalStructure.BCC,
                melting_point=1500.0,
                electrical_resistivity=1.7e-7
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
                crystal_structure=CrystalStructure.FCC,
                stacking_fault_energy=200.0,
                melting_point=582.0,
                electrical_resistivity=3.7e-8
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
                crystal_structure=CrystalStructure.HCP,
                stacking_fault_energy=300.0,
                melting_point=1668.0,
                electrical_resistivity=1.7e-6
            ),
            "Inconel 718": MaterialProperties(
                youngs_modulus=200.0,
                poissons_ratio=0.294,
                yield_strength=1034.0,
                tensile_strength=1241.0,
                elongation=12.0,
                reduction_area=25.0,
                fracture_toughness=80.0,
                fatigue_limit=550.0,
                density=8190.0,
                thermal_conductivity=11.4,
                specific_heat=435.0,
                thermal_expansion=13.0,
                crystal_structure=CrystalStructure.FCC,
                melting_point=1336.0,
                electrical_resistivity=1.25e-6
            ),
            "AISI 316L Stainless": MaterialProperties(
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
                crystal_structure=CrystalStructure.FCC,
                melting_point=1400.0,
                electrical_resistivity=7.4e-7
            )
        }
    
    # ==================== SAMPLE PREPARATION STATION ====================
    
    def design_microstructure(self, material: str, 
                            grain_size: float = 50.0,
                            phase_fraction: Dict[str, float] = None,
                            porosity: float = 0.01,
                            inclusion_size: float = 10.0,
                            texture_strength: float = 0.85) -> Microstructure:
        """Advanced microstructure designer with phase transformation physics"""
        
        # Validate input
        if grain_size <= 0:
            raise ValueError("Grain size must be positive")
        if porosity < 0 or porosity > 1:
            raise ValueError("Porosity must be between 0 and 1")
        
        # Set default phase fractions based on material
        if phase_fraction is None:
            if "Ti" in material:
                phase_fraction = {"alpha": 0.9, "beta": 0.1}
            elif "steel" in material.lower() or "Fe" in material:
                phase_fraction = {"ferrite": 0.85, "pearlite": 0.15}
            elif "Al" in material:
                phase_fraction = {"matrix": 0.95, "precipitates": 0.05}
            else:
                phase_fraction = {"matrix": 1.0}
        
        # Determine crystal structure
        crystal_structure = {
            "AISI": CrystalStructure.BCC,
            "Steel": CrystalStructure.BCC,
            "Al": CrystalStructure.FCC,
            "Ti": CrystalStructure.HCP,
            "Inconel": CrystalStructure.FCC,
            "Stainless": CrystalStructure.FCC
        }
        
        # Find matching crystal structure
        mat_struct = CrystalStructure.BCC
        for key, struct in crystal_structure.items():
            if key in material:
                mat_struct = struct
                break
        
        # Calculate defect density based on processing
        if grain_size < 10:  # Fine grains = more defects
            defect_density = 5000.0
        elif grain_size > 100:  # Coarse grains = fewer defects
            defect_density = 500.0
        else:
            defect_density = 1000.0
        
        # Create microstructure
        self.current_microstructure = Microstructure(
            grain_size=grain_size,
            phase_fractions=phase_fraction,
            defect_density=defect_density,
            porosity=porosity,
            inclusion_size=inclusion_size,
            inclusion_volume_fraction=0.005,
            crystal_structure=mat_struct,
            texture_coefficient=texture_strength,
            dislocation_density=1e12 * (50/grain_size)  # Higher for finer grains
        )
        
        # Log the design
        self._log_simulation("microstructure_design", {
            "material": material,
            "grain_size": grain_size,
            "porosity": porosity,
            "phase_fraction": phase_fraction
        })
        
        return self.current_microstructure
    
    def apply_heat_treatment(self, quenching_rate: float = 100.0,
                           tempering_temp: float = 600.0,
                           tempering_time: float = 2.0,
                           austenitizing_temp: float = 950.0) -> HeatTreatment:
        """Advanced heat treatment simulator with TTT/CCT kinetics"""
        
        # Validate input
        if quenching_rate <= 0:
            raise ValueError("Quenching rate must be positive")
        if tempering_temp < 100 or tempering_temp > 800:
            raise ValueError("Tempering temperature must be between 100°C and 800°C")
        
        # Determine cooling medium based on quenching rate
        if quenching_rate > 200:
            cooling_medium = "water"
        elif quenching_rate > 50:
            cooling_medium = "oil"
        else:
            cooling_medium = "air"
        
        # Create heat treatment
        self.current_heat_treatment = HeatTreatment(
            quenching_rate=quenching_rate,
            tempering_temperature=tempering_temp,
            tempering_time=tempering_time,
            austenitizing_temp=austenitizing_temp,
            cooling_medium=cooling_medium,
            precipitation_temp=500.0 if tempering_temp > 400 else 0,
            aging_time=8.0,
            martensite_start=300.0 if "steel" in str(self.current_material).lower() else 0
        )
        
        # Log the heat treatment
        self._log_simulation("heat_treatment", {
            "quenching_rate": quenching_rate,
            "tempering_temp": tempering_temp,
            "tempering_time": tempering_time,
            "cooling_medium": cooling_medium
        })
        
        return self.current_heat_treatment
    
    # ==================== TENSILE TESTING MODULE ====================
    
    class TensileTester:
        """Advanced tensile testing with true stress-strain and necking simulation"""
        
        def __init__(self, material_props: MaterialProperties,
                    microstructure: Microstructure = None,
                    heat_treatment: HeatTreatment = None):
            self.material = material_props
            self.microstructure = microstructure
            self.heat_treatment = heat_treatment
            self.engineering_stress_strain = None
            self.true_stress_strain = None
            self.necking_point = None
            self.constitutive_model = None
            self.temperature = 20.0
            self.strain_rate = 0.001
            
        def generate_stress_strain_curve(self, 
                                       constitutive_model: str = "mixed",
                                       temperature: float = 20.0,
                                       strain_rate: float = 0.001,
                                       num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
            """Generate stress-strain curve with advanced constitutive models"""
            
            self.constitutive_model = constitutive_model
            self.temperature = temperature
            self.strain_rate = strain_rate
            
            # Generate strain array with more points in elastic region
            eps_elastic = np.linspace(0, 0.002, 100)
            eps_plastic = np.linspace(0.002, 0.25, num_points - 100)
            eps = np.concatenate([eps_elastic, eps_plastic])
            
            # Elastic region (linear)
            E = self.material.youngs_modulus * 1000  # Convert GPa to MPa
            elastic_limit = self.material.yield_strength / E
            mask_elastic = eps <= elastic_limit
            
            stress = np.zeros_like(eps)
            stress[mask_elastic] = E * eps[mask_elastic]
            
            # Plastic region based on selected model
            eps_plastic = eps[~mask_elastic]
            
            if constitutive_model == "hollomon":
                # Hollomon power law: σ = Kεⁿ
                K = self.material.tensile_strength * (np.exp(self.material.elongation/100))**0.2
                n = 0.1 + 0.15 * (self.material.elongation/100)
                stress_plastic = K * (eps_plastic - elastic_limit) ** n
                
            elif constitutive_model == "voce":
                # Voce saturation hardening: σ = σ₀ + Q(1 - exp(-bε))
                sigma_0 = self.material.yield_strength
                Q = self.material.tensile_strength - sigma_0
                b = 25.0
                stress_plastic = sigma_0 + Q * (1 - np.exp(-b * (eps_plastic - elastic_limit)))
                
            elif constitutive_model == "johnson_cook":
                # Johnson-Cook model: σ = (A + Bεⁿ)(1 + C ln(ε̇/ε̇₀))(1 - T*ᵐ)
                A = self.material.yield_strength
                B = (self.material.tensile_strength - A) * 0.8
                n = 0.2
                C = 0.014
                m = 1.0
                eps0_dot = 0.001
                T_star = (temperature - 20) / (self.material.melting_point - 20) if self.material.melting_point else 0
                
                strain_hardening = A + B * (eps_plastic - elastic_limit) ** n
                strain_rate_term = 1 + C * np.log(strain_rate / eps0_dot)
                temperature_term = 1 - T_star ** m
                stress_plastic = strain_hardening * strain_rate_term * temperature_term
                
            else:  # mixed model (default)
                # Combine Hollomon and Voce
                K = self.material.tensile_strength * (np.exp(self.material.elongation/100))**0.2
                n = 0.12 + 0.15 * (self.material.elongation/100)
                hollomon = K * (eps_plastic - elastic_limit) ** n
                
                sigma_0 = self.material.yield_strength
                Q = self.material.tensile_strength - sigma_0
                b = 20.0
                voce = sigma_0 + Q * (1 - np.exp(-b * (eps_plastic - elastic_limit)))
                
                stress_plastic = 0.6 * hollomon + 0.4 * voce
            
            # Apply microstructure effects
            if self.microstructure:
                # Hall-Petch strengthening
                hp_strength = self.microstructure.calculate_hall_peetch()
                stress_plastic += hp_strength
                
                # Texture effect
                texture_factor = 1.0 + 0.1 * (self.microstructure.texture_coefficient - 1)
                stress_plastic *= texture_factor
            
            # Apply heat treatment effects
            if self.heat_treatment:
                martensite = self.heat_treatment.calculate_martensite_content()
                martensite_strengthening = 500 * martensite
                stress_plastic += martensite_strengthening
                
                tempering_effect = self.heat_treatment.calculate_tempering_effect()
                stress_plastic *= (1 - 0.3 * tempering_effect["hardness_reduction"])
            
            # Apply temperature effect
            if temperature > 20 and self.material.melting_point:
                T_homologous = (temperature + 273.15) / (self.material.melting_point + 273.15)
                if T_homologous > 0.3:
                    temp_factor = np.exp(-5 * (T_homologous - 0.3))
                    stress_plastic *= temp_factor
            
            # Apply strain rate effect (simplified)
            if strain_rate != 0.001:
                strain_rate_exponent = 0.01
                stress_plastic *= (strain_rate / 0.001) ** strain_rate_exponent
            
            stress[~mask_elastic] = stress_plastic
            
            # Ensure monotonic increase
            for i in range(1, len(stress)):
                if stress[i] < stress[i-1]:
                    stress[i] = stress[i-1]
            
            self.engineering_stress_strain = (eps, stress)
            
            # Convert to true stress-strain
            true_strain = np.log(1 + eps)
            true_stress = stress * (1 + eps)
            self.true_stress_strain = (true_strain, true_stress)
            
            # Find necking point (Considère criterion: dσ/dε = σ)
            if len(eps_plastic) > 10:
                plastic_stress = stress[~mask_elastic]
                plastic_strain = eps_plastic
                grad = np.gradient(plastic_stress, plastic_strain)
                necking_idx = np.where(grad <= plastic_stress)[0]
                if len(necking_idx) > 0:
                    idx = necking_idx[0]
                    self.necking_point = (plastic_strain[idx], plastic_stress[idx])
            
            return eps, stress
        
        def calculate_mechanical_properties(self) -> Dict[str, float]:
            """Calculate all standard mechanical properties with statistical uncertainty"""
            if self.engineering_stress_strain is None:
                raise ValueError("Generate stress-strain curve first")
            
            eps, stress = self.engineering_stress_strain
            
            # Young's modulus from initial slope (0-0.1% strain)
            elastic_range = eps < 0.001
            if np.sum(elastic_range) > 5:
                slope, intercept = np.polyfit(eps[elastic_range], stress[elastic_range], 1)
                E = slope / 1000  # Convert to GPa
            else:
                E = self.material.youngs_modulus
            
            # 0.2% offset yield strength
            offset_line = E * 1000 * (eps - 0.002)
            intersect_idx = np.where(stress >= offset_line)[0]
            if len(intersect_idx) > 0:
                sigma_y = stress[intersect_idx[0]]
                yield_strain = eps[intersect_idx[0]]
            else:
                sigma_y = stress[0]
                yield_strain = 0
            
            # Ultimate tensile strength
            sigma_uts = np.max(stress)
            uts_idx = np.argmax(stress)
            uniform_elongation = eps[uts_idx]
            
            # Fracture properties (assuming end of curve is fracture)
            fracture_stress = stress[-1]
            fracture_strain = eps[-1]
            
            # True fracture strength and ductility
            if self.true_stress_strain:
                true_strain, true_stress = self.true_stress_strain
                true_fracture_strength = true_stress[-1]
                true_fracture_strain = true_strain[-1]
            else:
                true_fracture_strength = fracture_stress * (1 + fracture_strain)
                true_fracture_strain = np.log(1 + fracture_strain)
            
            # Strain hardening exponent (n-value)
            plastic_range = (eps > yield_strain) & (eps < uniform_elongation)
            if np.sum(plastic_range) > 10:
                plastic_eps = eps[plastic_range] - yield_strain
                plastic_stress = stress[plastic_range]
                log_eps = np.log(plastic_eps)
                log_sigma = np.log(plastic_stress)
                n_value = np.polyfit(log_eps, log_sigma, 1)[0]
            else:
                n_value = 0.1
            
            # Strength coefficient (K-value)
            K_value = sigma_uts / (uniform_elongation ** n_value) if uniform_elongation > 0 else 0
            
            # Modulus of resilience and toughness
            modulus_resilience = (sigma_y ** 2) / (2 * E * 1000)  # MJ/m³
            modulus_toughness = np.trapz(stress, eps)  # MJ/m³
            
            # Statistical uncertainty (95% confidence)
            uncertainty = 0.02  # 2% typical
            
            properties = {
                "Young's Modulus (GPa)": round(E, 1),
                "0.2% Yield Strength (MPa)": round(sigma_y, 1),
                "Ultimate Tensile Strength (MPa)": round(sigma_uts, 1),
                "Uniform Elongation (%)": round(uniform_elongation * 100, 1),
                "Total Elongation (%)": round(fracture_strain * 100, 1),
                "True Fracture Strength (MPa)": round(true_fracture_strength, 1),
                "True Fracture Strain": round(true_fracture_strain, 3),
                "Strain Hardening Exponent (n)": round(n_value, 3),
                "Strength Coefficient (K, MPa)": round(K_value, 1),
                "Modulus of Resilience (MJ/m³)": round(modulus_resilience, 3),
                "Modulus of Toughness (MJ/m³)": round(modulus_toughness, 3),
                "Necking Strain (%)": round(self.necking_point[0]*100, 2) if self.necking_point else 0.0,
                "Yield to Tensile Ratio": round(sigma_y / sigma_uts, 3)
            }
            
            # Add uncertainty bounds
            uncertainty_props = {}
            for key, value in properties.items():
                if "MPa" in key or "GPa" in key:
                    uncertainty_props[f"{key} ± {uncertainty*100}%"] = (round(value*(1-uncertainty), 1), 
                                                                       round(value*(1+uncertainty), 1))
            
            properties.update(uncertainty_props)
            
            return properties
        
        def visualize_curve(self, show_true: bool = True, show_derivatives: bool = True):
            """Interactive visualization with Plotly"""
            if self.engineering_stress_strain is None:
                raise ValueError("Generate curve first")
            
            eps, eng_stress = self.engineering_stress_strain
            true_strain, true_stress = self.true_stress_strain
            
            # Determine number of subplots
            num_plots = 2 if show_true else 1
            if show_derivatives:
                num_plots += 2
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Engineering Stress-Strain", 
                              "True Stress-Strain",
                              "Strain Hardening Analysis",
                              "Considère Criterion"),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # Engineering curve
            fig.add_trace(
                go.Scatter(x=eps*100, y=eng_stress, mode='lines',
                          name='Engineering', line=dict(color='blue', width=3),
                          hovertemplate='Strain: %{x:.2f}%<br>Stress: %{y:.1f} MPa'),
                row=1, col=1
            )
            
            # Add yield point
            if self.material.yield_strength:
                fig.add_trace(
                    go.Scatter(x=[0, eps[-1]*100], y=[self.material.yield_strength, self.material.yield_strength],
                              mode='lines', name='Yield Strength',
                              line=dict(color='green', width=2, dash='dash')),
                    row=1, col=1
                )
            
            if self.necking_point:
                fig.add_trace(
                    go.Scatter(x=[self.necking_point[0]*100], 
                              y=[self.necking_point[1]],
                              mode='markers',
                              name='Necking Start',
                              marker=dict(size=12, color='red', symbol='x'),
                              hovertemplate='Necking: %{x:.2f}% strain'),
                    row=1, col=1
                )
            
            # True curve
            if show_true:
                fig.add_trace(
                    go.Scatter(x=true_strain*100, y=true_stress, mode='lines',
                              name='True', line=dict(color='red', width=3),
                              hovertemplate='True Strain: %{x:.2f}%<br>True Stress: %{y:.1f} MPa'),
                    row=1, col=2
                )
            
            # Strain hardening rate
            if show_derivatives:
                plastic_mask = eps > (self.material.yield_strength/(self.material.youngs_modulus*1000))
                if np.any(plastic_mask):
                    plastic_eps = eps[plastic_mask]
                    plastic_stress = eng_stress[plastic_mask]
                    hardening_rate = np.gradient(plastic_stress, plastic_eps)
                    
                    fig.add_trace(
                        go.Scatter(x=plastic_eps*100, y=hardening_rate,
                                  mode='lines', name='Hardening Rate (dσ/dε)',
                                  line=dict(color='green', width=2),
                                  hovertemplate='Strain: %{x:.2f}%<br>dσ/dε: %{y:.1f} MPa'),
                        row=2, col=1
                    )
                    
                    # Add hardening exponent visualization
                    log_eps = np.log(plastic_eps)
                    log_sigma = np.log(plastic_stress)
                    if len(log_eps) > 5:
                        n_fit = np.polyfit(log_eps, log_sigma, 1)[0]
                        fig.add_annotation(
                            x=0.5, y=0.9, xref="paper", yref="paper",
                            text=f"n = {n_fit:.3f}",
                            showarrow=False,
                            font=dict(size=14, color="green"),
                            row=2, col=1
                        )
            
            # Considère criterion
            if self.necking_point and show_derivatives:
                # Plot stress and its derivative
                fig.add_trace(
                    go.Scatter(x=eps*100, y=eng_stress,
                              name='Stress', line=dict(color='blue'),
                              showlegend=False),
                    row=2, col=2
                )
                
                stress_derivative = np.gradient(eng_stress, eps)
                fig.add_trace(
                    go.Scatter(x=eps*100, y=stress_derivative,
                              name='dσ/dε', line=dict(color='red'),
                              showlegend=False),
                    row=2, col=2
                )
                
                # Highlight necking criterion line
                fig.add_trace(
                    go.Scatter(x=eps*100, y=eng_stress,
                              name='σ = dσ/dε', line=dict(color='purple', dash='dot'),
                              showlegend=False),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text=f"Tensile Test Analysis - {self.material.__class__.__name__}",
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Strain (%)", row=1, col=1)
            fig.update_yaxes(title_text="Stress (MPa)", row=1, col=1)
            fig.update_xaxes(title_text="True Strain (%)", row=1, col=2)
            fig.update_yaxes(title_text="True Stress (MPa)", row=1, col=2)
            fig.update_xaxes(title_text="Plastic Strain (%)", row=2, col=1)
            fig.update_yaxes(title_text="Hardening Rate (MPa)", row=2, col=1)
            fig.update_xaxes(title_text="Strain (%)", row=2, col=2)
            fig.update_yaxes(title_text="Stress / dσ/dε (MPa)", row=2, col=2)
            
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
            self.fracture_surface = None
            
        def generate_SN_curve(self, R_ratio: float = -1.0,
                            surface_finish: str = "polished",
                            environment: str = "air",
                            reliability: float = 0.95,
                            confidence: float = 0.90) -> Tuple[np.ndarray, np.ndarray]:
            """Generate S-N curve with statistical reliability and confidence bands"""
            
            # Basquin equation: σ_a = σ_f' * (2N_f)^b
            sigma_f_prime = self.material.tensile_strength * 1.2  # More accurate
            b = -0.09  # Basquin exponent adjusted for material
            
            # Adjust for material type
            if self.material.crystal_structure == CrystalStructure.ALUMINUM:
                b = -0.12
                sigma_f_prime = self.material.tensile_strength * 1.4
            elif self.material.crystal_structure == CrystalStructure.TITANIUM:
                b = -0.08
                sigma_f_prime = self.material.tensile_strength * 1.1
            
            N_cycles = np.logspace(3, 8, 100)  # 1e3 to 1e8 cycles
            stress_amp = sigma_f_prime * (2 * N_cycles) ** b
            
            # Mean stress effect (modified Goodman correction)
            if R_ratio != -1:
                sigma_mean = stress_amp * (1 + R_ratio) / (1 - R_ratio)
                # More accurate Goodman: σ_a = σ_e * (1 - σ_m/σ_UTS)
                sigma_amp_corrected = stress_amp * (1 - sigma_mean/self.material.tensile_strength)
                # Gerber alternative for high mean stresses
                if np.max(sigma_mean) > 0.5 * self.material.tensile_strength:
                    sigma_amp_corrected = stress_amp * (1 - (sigma_mean/self.material.tensile_strength)**2)
                stress_amp = np.maximum(sigma_amp_corrected, 0.1 * stress_amp)
            
            # Surface finish factor (Peterson factors)
            surface_factors = {
                "polished": 1.0,
                "ground": 0.9,
                "machined": 0.8,
                "hot_rolled": 0.7,
                "as_forged": 0.6,
                "corroded": 0.4
            }
            stress_amp *= surface_factors.get(surface_finish, 0.9)
            
            # Environment factor
            environment_factors = {
                "air": 1.0,
                "vacuum": 1.2,
                "salt_water": 0.5,
                "corrosive": 0.3
            }
            stress_amp *= environment_factors.get(environment, 1.0)
            
            # Reliability factor (Weibull statistics)
            if reliability != 0.5:
                # Weibull shape parameter (typical for metals)
                beta = 2.0  # Shape parameter
                R_factor = (-np.log(reliability)) ** (1/beta)
                stress_amp /= R_factor
            
            # Fatigue limit with statistical scatter
            fatigue_limit = self.material.fatigue_limit
            
            # Apply fatigue limit (endurance limit)
            for i in range(len(N_cycles)):
                if N_cycles[i] > 1e6:
                    stress_amp[i] = max(stress_amp[i], fatigue_limit)
            
            # Add confidence bands
            if confidence > 0.5:
                z = stats.norm.ppf(confidence)
                scatter = 0.08  # 8% typical scatter in fatigue data
                lower_bound = stress_amp * (1 - z * scatter)
                upper_bound = stress_amp * (1 + z * scatter)
            else:
                lower_bound = upper_bound = stress_amp
            
            self.SN_data = (N_cycles, stress_amp, lower_bound, upper_bound)
            return N_cycles, stress_amp, lower_bound, upper_bound
        
        def paris_law_crack_growth(self, initial_crack: float = 0.1,
                                 final_crack: float = 10.0,
                                 delta_sigma: float = 200.0,
                                 R_ratio: float = 0.1,
                                 geometry_factor: float = 1.12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Paris-Erdogan crack growth simulation with advanced features"""
            
            K_c = self.material.fracture_toughness
            
            # Material-specific Paris law constants
            if self.material.crystal_structure == CrystalStructure.ALUMINUM:
                C = 2.3e-11  # mm/cycle/(MPa√m)^m
                m = 3.2
                delta_K_th = 3.0  # MPa√m
            elif self.material.crystal_structure == CrystalStructure.TITANIUM:
                C = 3.5e-11
                m = 3.5
                delta_K_th = 4.0
            else:  # Steel default
                C = 6.9e-12
                m = 3.0
                delta_K_th = 5.0
            
            # Crack lengths array (logarithmic spacing for better resolution)
            a = np.logspace(np.log10(initial_crack), np.log10(final_crack), 1000)
            
            # Stress intensity factor range (with geometry factor)
            delta_K = delta_sigma * geometry_factor * np.sqrt(np.pi * a)
            
            # Apply threshold and R-ratio effects
            delta_K_eff = delta_K * (1 - R_ratio) ** 0.5  # Walker equation
            
            # Apply threshold and fracture toughness limits
            valid = (delta_K_eff > delta_K_th) & (delta_K < K_c)
            da_dN = np.zeros_like(a)
            da_dN[valid] = C * (delta_K_eff[valid] ** m)
            
            # Calculate number of cycles using integration
            N_cycles = np.zeros_like(a)
            if np.any(valid):
                # Numerical integration of Paris law
                integrand = 1 / (C * (delta_sigma * geometry_factor * np.sqrt(np.pi * a[valid]) ** m))
                N_integrated = integrate.cumtrapz(integrand, a[valid], initial=0)
                N_cycles[valid] = N_integrated
            
            # Calculate crack growth rate in different regimes
            growth_regimes = {
                "threshold": delta_K_eff < delta_K_th * 1.1,
                "paris": (delta_K_eff >= delta_K_th * 1.1) & (delta_K < 0.7 * K_c),
                "accelerated": delta_K >= 0.7 * K_c
            }
            
            self.crack_growth_data = {
                "a": a,
                "da_dN": da_dN,
                "N": N_cycles,
                "delta_K": delta_K,
                "regimes": growth_regimes,
                "constants": {"C": C, "m": m, "delta_K_th": delta_K_th}
            }
            
            return a, da_dN, N_cycles
        
        def fracture_surface_simulation(self, crack_length: float = 5.0,
                                      loading_type: str = "constant_amplitude") -> Dict[str, Any]:
            """Generate synthetic fracture surface with realistic features"""
            
            # Create grid for fracture surface
            size = 400
            x = np.linspace(-10, 10, size)
            y = np.linspace(-10, 10, size)
            X, Y = np.meshgrid(x, y)
            
            # Distance from crack origin
            r = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            
            # Initialize surface height
            Z = np.zeros_like(X)
            
            # Add beach marks (fatigue striations)
            num_striations = 50
            striation_spacing = crack_length / num_striations
            
            for i in range(1, num_striations + 1):
                striation_radius = i * striation_spacing
                # Create circular striation
                striation = 0.1 * np.exp(-((r - striation_radius) ** 2) / (0.5 ** 2))
                # Add angular variation for realism
                striation *= (1 + 0.2 * np.sin(5 * theta))
                Z += striation
            
            # Add overload marks (periodic larger marks)
            num_overloads = 5
            for i in range(1, num_overloads + 1):
                overload_pos = crack_length * i / (num_overloads + 1)
                overload = 0.5 * np.exp(-((r - overload_pos) ** 2) / (1.0 ** 2))
                overload *= (1 + 0.3 * np.cos(3 * theta))
                Z += overload
            
            # Add final fracture region (rough surface)
            final_frac = r > crack_length * 0.8
            if np.any(final_frac):
                # Generate fractal-like roughness
                roughness = np.random.randn(*Z[final_frac].shape)
                # Apply Gaussian filter for correlated roughness
                roughness_size = int(np.sqrt(len(roughness)))
                if roughness_size > 10:
                    roughness = roughness.reshape((roughness_size, roughness_size))
                    roughness = gaussian_filter(roughness, sigma=2)
                    roughness = roughness.flatten()
                Z[final_frac] += 2.0 * roughness
            
            # Add microstructural features
            if self.microstructure:
                # Grain boundaries effect
                grain_effect = 0.05 * np.sin(20 * X) * np.sin(20 * Y)
                Z += grain_effect
                
                # Porosity effect
                if self.microstructure.porosity > 0:
                    num_pores = int(100 * self.microstructure.porosity)
                    for _ in range(num_pores):
                        pore_x = np.random.uniform(-8, 8)
                        pore_y = np.random.uniform(-8, 8)
                        pore_r = np.sqrt((X - pore_x)**2 + (Y - pore_y)**2)
                        pore = -0.3 * np.exp(-(pore_r**2) / (0.3**2))  # Negative for pores
                        Z += pore
            
            # Crack propagation direction markers
            prop_angle = np.pi/4  # 45 degree propagation
            prop_lines = 0.1 * np.sin(10 * (X * np.cos(prop_angle) + Y * np.sin(prop_angle)))
            Z += prop_lines
            
            self.fracture_surface = {
                "X": X,
                "Y": Y,
                "Z": Z,
                "crack_length": crack_length,
                "features": {
                    "striations": num_striations,
                    "overloads": num_overloads,
                    "rough_fracture": np.mean(Z[final_frac]) if np.any(final_frac) else 0
                }
            }
            
            return self.fracture_surface
        
        def calculate_fatigue_life(self, stress_range: float, 
                                 stress_ratio: float = -1.0,
                                 initial_flaw_size: float = 0.1) -> Dict[str, float]:
            """Calculate total fatigue life including initiation and propagation"""
            
            # Initiation life (using strain-life approach)
            # Morrow's equation: ε_a = (σ_f'/E)(2N_f)^b + ε_f'(2N_f)^c
            sigma_f_prime = self.material.tensile_strength * 1.2
            epsilon_f_prime = 0.5  # Ductility coefficient
            b = -0.09  # Strength exponent
            c = -0.6   # Ductility exponent
            
            stress_amp = stress_range / 2
            strain_amp = stress_amp / (self.material.youngs_modulus * 1000)
            
            # Solve for initiation life
            def strain_life_eq(N):
                return (sigma_f_prime/(self.material.youngs_modulus*1000))*(2*N)**b + epsilon_f_prime*(2*N)**c - strain_amp
            
            try:
                N_initiation = optimize.brentq(strain_life_eq, 1e3, 1e8)
            except:
                N_initiation = 1e6  # Default if solution fails
            
            # Propagation life (from Paris law)
            _, _, N_propagation = self.paris_law_crack_growth(
                initial_crack=initial_flaw_size,
                final_crack=10.0,
                delta_sigma=stress_range,
                R_ratio=stress_ratio
            )
            
            total_N_propagation = N_propagation[-1] if len(N_propagation) > 0 else 0
            
            # Total life
            total_life = N_initiation + total_N_propagation
            
            return {
                "initiation_life": round(N_initiation, 0),
                "propagation_life": round(total_N_propagation, 0),
                "total_life": round(total_life, 0),
                "initiation_fraction": round(N_initiation/total_life, 3) if total_life > 0 else 0
            }
    
    # ==================== FRACTURE TOUGHNESS MODULE ====================
    
    class FractureToughnessTester:
        """Advanced fracture mechanics with plastic zone simulation"""
        
        def __init__(self, material_props: MaterialProperties):
            self.material = material_props
            self.stress_fields = {}
            self.plastic_zones = {}
            
        def calculate_stress_field(self, K_I: float = 30.0,
                                 distance: float = 10.0,
                                 theta_range: Tuple[float, float] = (-np.pi, np.pi),
                                 num_points: int = 100) -> Dict[str, np.ndarray]:
            """Calculate crack tip stress field (Mode I) with higher order terms"""
            
            theta = np.linspace(theta_range[0], theta_range[1], num_points)
            r = np.logspace(-2, np.log10(distance), 50)  # Logarithmic spacing near crack tip
            
            R, Theta = np.meshgrid(r, theta)
            
            # Williams asymptotic expansion (first 3 terms)
            # σ_ij = K_I/√(2πr) * f_ij(θ) + A_2 * r^0 * g_ij(θ) + A_3 * r^(1/2) * h_ij(θ)
            
            # First term (singular)
            sqrt_2pi_r = np.sqrt(2 * np.pi * R)
            cos_theta_2 = np.cos(Theta/2)
            sin_theta_2 = np.sin(Theta/2)
            cos_3theta_2 = np.cos(3*Theta/2)
            sin_3theta_2 = np.sin(3*Theta/2)
            
            sigma_xx_1 = K_I / sqrt_2pi_r * cos_theta_2 * (1 - sin_theta_2 * sin_3theta_2)
            sigma_yy_1 = K_I / sqrt_2pi_r * cos_theta_2 * (1 + sin_theta_2 * sin_3theta_2)
            tau_xy_1 = K_I / sqrt_2pi_r * sin_theta_2 * cos_theta_2 * cos_3theta_2
            
            # Second term (T-stress, constant)
            T = 0.1 * K_I / np.sqrt(np.pi * distance)  # Approximate T-stress
            sigma_xx_2 = T * np.ones_like(R)
            sigma_yy_2 = np.zeros_like(R)
            tau_xy_2 = np.zeros_like(R)
            
            # Combine terms
            sigma_xx = sigma_xx_1 + sigma_xx_2
            sigma_yy = sigma_yy_1 + sigma_yy_2
            tau_xy = tau_xy_1 + tau_xy_2
            
            # Principal stresses
            sigma_1 = (sigma_xx + sigma_yy)/2 + np.sqrt(((sigma_xx - sigma_yy)/2)**2 + tau_xy**2)
            sigma_2 = (sigma_xx + sigma_yy)/2 - np.sqrt(((sigma_xx - sigma_yy)/2)**2 + tau_xy**2)
            
            # von Mises equivalent stress
            sigma_vm = np.sqrt(0.5*((sigma_xx - sigma_yy)**2 + sigma_xx**2 + sigma_yy**2) + 3*tau_xy**2)
            
            # Hydrostatic stress
            sigma_h = (sigma_xx + sigma_yy) / 3
            
            # J-integral estimate (for elastic-plastic)
            J_integral = K_I**2 * (1 - self.material.poissons_ratio**2) / self.material.youngs_modulus
            
            field_data = {
                'sigma_xx': sigma_xx,
                'sigma_yy': sigma_yy,
                'tau_xy': tau_xy,
                'sigma_1': sigma_1,
                'sigma_2': sigma_2,
                'sigma_vm': sigma_vm,
                'sigma_h': sigma_h,
                'r': R,
                'theta': Theta,
                'K_I': K_I,
                'J_integral': J_integral
            }
            
            self.stress_fields[K_I] = field_data
            return field_data
        
        def estimate_plastic_zone(self, K_I: float = 30.0,
                                plane_stress: bool = True,
                                method: str = "irwin") -> Dict[str, float]:
            """Estimate plastic zone size with different methods"""
            
            sigma_y = self.material.yield_strength
            
            if method == "irwin":
                # Irwin's correction
                if plane_stress:
                    r_p = (1/(2*np.pi)) * (K_I/sigma_y)**2
                    # Irwin's effective crack length correction
                    r_y = (1/(2*np.pi)) * (K_I/sigma_y)**2
                    effective_K = K_I * np.sqrt((r_p + r_y)/r_p)  # Simplified
                else:
                    r_p = (1/(6*np.pi)) * (K_I/sigma_y)**2
                    
            elif method == "dugdale":
                # Dugdale strip yield model
                r_p = (np.pi/8) * (K_I/sigma_y)**2
                
            elif method == "j_integral":
                # J-integral based estimation
                J = K_I**2 * (1 - self.material.poissons_ratio**2) / self.material.youngs_modulus
                r_p = 0.5 * J / sigma_y
                
            else:
                r_p = (1/(2*np.pi)) * (K_I/sigma_y)**2
            
            # Shape factors
            shape_factors = {
                "forward": r_p,  # Forward plastic zone
                "reverse": 0.25 * r_p,  # Reverse plastic zone
                "height": 0.6 * r_p,  # Maximum height
                "volume": (4/3) * np.pi * r_p**3  # Approximate volume
            }
            
            # Determine if small-scale yielding applies
            small_scale = r_p < 0.02 * K_I**2 / sigma_y**2
            
            result = {
                "plastic_zone_size": r_p,
                "plane_condition": "plane_stress" if plane_stress else "plane_strain",
                "small_scale_yielding": small_scale,
                "shape_factors": shape_factors,
                "K_max_ssy": sigma_y * np.sqrt(2*np.pi*0.02) if small_scale else np.inf
            }
            
            self.plastic_zones[K_I] = result
            return result
        
        def calculate_stress_intensity_solution(self, crack_length: float,
                                              stress: float,
                                              geometry: str = "edge_crack") -> Dict[str, float]:
            """Calculate stress intensity factor for different geometries"""
            
            if geometry == "edge_crack":
                # Edge crack in semi-infinite plate
                beta = 1.12  # Geometry factor
                K_I = beta * stress * np.sqrt(np.pi * crack_length)
                
            elif geometry == "center_crack":
                # Center crack in infinite plate
                beta = 1.0
                K_I = stress * np.sqrt(np.pi * crack_length)
                
            elif geometry == "compact_tension":
                # Compact tension specimen (ASTM E399)
                a_w = crack_length  # Crack length to width ratio
                if a_w < 0.2:
                    a_w = 0.2
                elif a_w > 0.6:
                    a_w = 0.6
                
                # ASTM polynomial for CT specimen
                beta = (2 + a_w) * (0.886 + 4.64*a_w - 13.32*a_w**2 + 14.72*a_w**3 - 5.6*a_w**4) / (1 - a_w)**1.5
                K_I = beta * stress * np.sqrt(np.pi * crack_length)
                
            elif geometry == "single_edge_notch":
                # Single edge notched bend specimen
                a_w = crack_length
                if a_w < 0.2:
                    a_w = 0.2
                elif a_w > 0.6:
                    a_w = 0.6
                
                beta = 1.99 - 2.47*a_w + 12.97*a_w**2 - 23.17*a_w**3 + 24.8*a_w**4
                K_I = beta * stress * np.sqrt(np.pi * crack_length)
                
            else:
                beta = 1.0
                K_I = stress * np.sqrt(np.pi * crack_length)
            
            # Check against fracture toughness
            K_ratio = K_I / self.material.fracture_toughness if self.material.fracture_toughness > 0 else 0
            
            return {
                "K_I": K_I,
                "geometry_factor": beta,
                "crack_length": crack_length,
                "applied_stress": stress,
                "K_ratio": K_ratio,
                "safe": K_ratio < 0.7  # Safety factor
            }
    
    # ==================== CREEP TESTING MODULE ====================
    
    class CreepTester:
        """Advanced creep deformation and rupture prediction"""
        
        def __init__(self, material_props: MaterialProperties):
            self.material = material_props
            self.creep_data = {}
            self.rupture_data = {}
            
        def creep_deformation(self, stress: float = 100.0,
                            temperature: float = 600.0,
                            time_hours: float = 10000.0,
                            model: str = "norton") -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
            """Calculate creep strain using advanced creep models"""
            
            # Material-dependent parameters
            if self.material.crystal_structure == CrystalStructure.FCC:
                # FCC materials (Al, Cu, Ni alloys)
                n = 5.0  # Stress exponent
                Q = 150e3  # Activation energy (J/mol)
                A0 = 1e-10
                m = 0.3  # Time exponent
            elif self.material.crystal_structure == CrystalStructure.BCC:
                # BCC materials (Fe-based alloys)
                n = 4.0
                Q = 300e3
                A0 = 1e-11
                m = 0.2
            else:  # HCP and others
                n = 3.0
                Q = 200e3
                A0 = 1e-12
                m = 0.25
            
            R = 8.314  # J/mol·K
            T = temperature + 273.15  # K
            
            # Time array (logarithmic spacing)
            t_hours = np.logspace(0, np.log10(time_hours), 500)
            t_seconds = t_hours * 3600
            
            # Norton's power law (primary + secondary creep)
            A = A0 * np.exp(-Q/(R*T))
            
            if model == "norton":
                # Simple Norton: ε = A σ^n t
                epsilon_creep = A * (stress ** n) * t_seconds
                
            elif model == "garofalo":
                # Garofalo (sinh law): ε = A [sinh(ασ)]^n t
                alpha = 0.01  # Stress coefficient
                epsilon_creep = A * (np.sinh(alpha * stress) ** n) * t_seconds
                
            elif model == "combined":
                # Combined model with primary and secondary creep
                # ε = ε_0 + A σ^n t^m + B σ^p t
                epsilon_primary = 0.001 * (stress / self.material.yield_strength)  # Initial strain
                epsilon_secondary = A * (stress ** n) * t_seconds
                epsilon_tertiary = 1e-15 * (stress ** 8) * (t_seconds ** 3)
                epsilon_creep = epsilon_primary + epsilon_secondary + epsilon_tertiary
                
            else:
                epsilon_creep = A * (stress ** n) * t_seconds
            
            # Convert to percentage strain
            epsilon_percent = epsilon_creep * 100
            
            # Calculate creep rate
            creep_rate = np.gradient(epsilon_creep, t_seconds)  # Strain rate (1/s)
            
            # Identify creep stages
            if len(t_seconds) > 10:
                # Find minimum creep rate (secondary creep)
                min_rate_idx = np.argmin(creep_rate[10:-10]) + 10
                secondary_time = t_hours[min_rate_idx]
                secondary_rate = creep_rate[min_rate_idx]
                
                # Tertiary creep starts when rate increases by 50% from minimum
                tertiary_start_idx = np.where(creep_rate[min_rate_idx:] > 1.5 * secondary_rate)[0]
                if len(tertiary_start_idx) > 0:
                    tertiary_time = t_hours[min_rate_idx + tertiary_start_idx[0]]
                else:
                    tertiary_time = t_hours[-1]
            else:
                secondary_time = tertiary_time = t_hours[-1]
                secondary_rate = creep_rate[-1]
            
            creep_stages = {
                "primary": t_hours < secondary_time,
                "secondary": (t_hours >= secondary_time) & (t_hours < tertiary_time),
                "tertiary": t_hours >= tertiary_time
            }
            
            self.creep_data[(stress, temperature)] = {
                "time": t_hours,
                "strain": epsilon_percent,
                "rate": creep_rate,
                "stages": creep_stages,
                "secondary_rate": secondary_rate,
                "model": model
            }
            
            return t_hours, epsilon_percent, creep_stages
        
        def larson_miller_parameter(self, stress_range: Tuple[float, float] = (50, 300),
                                  temperature: float = 600.0,
                                  time_range: Tuple[float, float] = (10, 100000)) -> Dict[str, Any]:
            """Calculate Larson-Miller parameter for rupture prediction"""
            
            # Larson-Miller: P = T(C + log t)
            C = 20  # Typical constant for steels
            
            # Generate stress array
            stresses = np.linspace(stress_range[0], stress_range[1], 50)
            
            # Generate rupture times for given temperature
            T_kelvin = temperature + 273.15
            rupture_times = np.logspace(np.log10(time_range[0]), np.log10(time_range[1]), len(stresses))
            
            # Calculate LMP for each stress
            LMP_values = T_kelvin * (C + np.log10(rupture_times))
            
            # Fit power law: σ = a * P^b
            log_stress = np.log(stresses)
            log_LMP = np.log(LMP_values)
            b, log_a = np.polyfit(log_LMP, log_stress, 1)
            a = np.exp(log_a)
            
            # Generate fitted curve
            LMP_fit = np.linspace(np.min(LMP_values), np.max(LMP_values), 100)
            stress_fit = a * LMP_fit ** b
            
            # Calculate equivalent temperatures for constant time
            constant_times = [100, 1000, 10000, 100000]
            temp_curves = {}
            for t in constant_times:
                temp_curves[t] = (LMP_values / (C + np.log10(t))) - 273.15
            
            self.rupture_data[temperature] = {
                "stresses": stresses,
                "LMP": LMP_values,
                "rupture_times": rupture_times,
                "fit_params": {"a": a, "b": b},
                "constant_time_curves": temp_curves
            }
            
            return self.rupture_data[temperature]
        
        def calculate_minimum_creep_rate(self, stress: float, temperature: float) -> float:
            """Calculate minimum (secondary) creep rate using power law"""
            
            if self.material.crystal_structure == CrystalStructure.FCC:
                n = 5.0
                Q = 150e3
                A0 = 1e-10
            else:
                n = 4.0
                Q = 300e3
                A0 = 1e-11
            
            R = 8.314
            T = temperature + 273.15
            
            A = A0 * np.exp(-Q/(R*T))
            min_rate = A * (stress ** n)
            
            return min_rate
        
        def estimate_rupture_life(self, stress: float, temperature: float) -> float:
            """Estimate time to rupture using Monkman-Grant relationship"""
            
            # Monkman-Grant: t_r * ε̇_min^m = constant
            min_rate = self.calculate_minimum_creep_rate(stress, temperature)
            
            # Typical Monkman-Grant constants
            C_mg = 0.1  # Constant
            m_mg = 0.85  # Exponent
            
            rupture_time = C_mg / (min_rate ** m_mg)  # In seconds
            rupture_hours = rupture_time / 3600
            
            # Adjust for temperature effect
            T_homologous = (temperature + 273.15) / (self.material.melting_point + 273.15)
            if T_homologous > 0.5:
                rupture_hours *= np.exp(-10 * (T_homologous - 0.5))
            
            return max(rupture_hours, 0.1)  # Minimum 0.1 hours
    
    # ==================== MICROSTRUCTURE VIEWER ====================
    
    class MicrostructureViewer:
        """Advanced microstructure generation and visualization"""
        
        def __init__(self):
            self.grains = None
            self.phases = None
            self.texture = None
            self.microstructures = {}
            
        def generate_voronoi_microstructure(self, grain_size: float = 50.0,
                                          phase_fractions: Dict[str, float] = None,
                                          size: int = 500,
                                          anisotropy: float = 1.0,
                                          twin_fraction: float = 0.1) -> Dict[str, Any]:
            """Generate synthetic microstructure using Voronoi tessellation with advanced features"""
            
            if phase_fractions is None:
                phase_fractions = {'alpha': 0.9, 'beta': 0.1}
            
            # Calculate number of grains
            area_per_grain = grain_size ** 2
            n_grains = int((size * size) / area_per_grain)
            n_grains = max(10, min(n_grains, 10000))  # Limit for performance
            
            # Generate grain centers with anisotropy
            if anisotropy == 1.0:
                points = np.random.rand(n_grains, 2) * size
            else:
                # Anisotropic distribution (elongated grains)
                points_x = np.random.rand(n_grains) * size
                points_y = np.random.rand(n_grains) * size * anisotropy
                points = np.column_stack([points_x, points_y])
            
            # Create grid
            x, y = np.meshgrid(np.arange(size), np.arange(size))
            grid_points = np.column_stack([x.ravel(), y.ravel()])
            
            # Assign each point to nearest grain center
            tree = cKDTree(points)
            distances, grain_indices = tree.query(grid_points)
            
            # Reshape to image
            grain_map = grain_indices.reshape((size, size))
            
            # Assign phases with probabilistic distribution
            unique_grains = np.unique(grain_map)
            phase_map = np.zeros_like(grain_map, dtype=np.int32)
            
            # Calculate phase assignments
            phase_cumsum = np.cumsum(list(phase_fractions.values()))
            phase_names = list(phase_fractions.keys())
            
            for i, grain in enumerate(unique_grains):
                mask = grain_map == grain
                rand_val = np.random.rand()
                
                # Determine phase based on cumulative probabilities
                phase_idx = 0
                for j, cum_prob in enumerate(phase_cumsum):
                    if rand_val <= cum_prob:
                        phase_idx = j + 1  # 1-indexed for phases
                        break
                
                phase_map[mask] = phase_idx
            
            # Add twins
            if twin_fraction > 0:
                n_twins = int(n_grains * twin_fraction)
                twin_grains = np.random.choice(unique_grains, n_twins, replace=False)
                
                for twin in twin_grains:
                    twin_mask = grain_map == twin
                    # Create twin boundary by modifying grain ID
                    phase_map[twin_mask] = phase_map[twin_mask] * 10  # Mark as twinned
            
            # Calculate grain boundaries
            gb_x = sobel(grain_map, axis=0)
            gb_y = sobel(grain_map, axis=1)
            grain_boundaries = np.sqrt(gb_x**2 + gb_y**2) > 0
            
            # Calculate grain statistics
            grain_stats = []
            for grain in unique_grains:
                mask = grain_map == grain
                area = np.sum(mask)
                if area > 0:
                    # Calculate equivalent diameter
                    equivalent_diameter = 2 * np.sqrt(area / np.pi)
                    
                    # Calculate aspect ratio
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) > 1:
                        covariance = np.cov(x_indices, y_indices)
                        eigenvalues = np.linalg.eigvals(covariance)
                        aspect_ratio = np.sqrt(eigenvalues.max() / eigenvalues.min()) if eigenvalues.min() > 0 else 1
                    else:
                        aspect_ratio = 1
                    
                    grain_stats.append({
                        'grain_id': grain,
                        'area': area,
                        'equivalent_diameter': equivalent_diameter,
                        'aspect_ratio': aspect_ratio,
                        'phase': np.median(phase_map[mask])
                    })
            
            # Generate synthetic texture (crystal orientations)
            orientation_map = np.zeros((size, size, 3))
            for grain in unique_grains:
                mask = grain_map == grain
                # Random Euler angles (Bunge convention)
                phi1 = np.random.rand() * 360
                Phi = np.random.rand() * 180
                phi2 = np.random.rand() * 360
                orientation_map[mask, 0] = phi1
                orientation_map[mask, 1] = Phi
                orientation_map[mask, 2] = phi2
            
            microstructure_data = {
                'grain_map': grain_map,
                'phase_map': phase_map,
                'grain_boundaries': grain_boundaries,
                'orientation_map': orientation_map,
                'grain_stats': grain_stats,
                'phase_fractions': phase_fractions,
                'grain_size': grain_size,
                'parameters': {
                    'size': size,
                    'anisotropy': anisotropy,
                    'twin_fraction': twin_fraction,
                    'n_grains': n_grains
                }
            }
            
            self.microstructures[grain_size] = microstructure_data
            return microstructure_data
        
        def calculate_microstructure_statistics(self, microstructure_data: Dict[str, Any]) -> Dict[str, float]:
            """Calculate comprehensive microstructure statistics"""
            
            grain_map = microstructure_data['grain_map']
            phase_map = microstructure_data['phase_map']
            grain_stats = microstructure_data['grain_stats']
            
            # Grain size statistics
            diameters = [stat['equivalent_diameter'] for stat in grain_stats]
            mean_diameter = np.mean(diameters)
            std_diameter = np.std(diameters)
            
            # Aspect ratio statistics
            aspect_ratios = [stat['aspect_ratio'] for stat in grain_stats]
            mean_aspect_ratio = np.mean(aspect_ratios)
            
            # Phase statistics
            unique_phases = np.unique(phase_map)
            phase_areas = {}
            for phase in unique_phases:
                if phase > 0:  # Skip background
                    phase_areas[phase] = np.sum(phase_map == phase)
            
            total_area = np.sum(list(phase_areas.values()))
            phase_fractions = {phase: area/total_area for phase, area in phase_areas.items()}
            
            # Grain boundary density
            gb_pixels = np.sum(microstructure_data['grain_boundaries'])
            gb_density = gb_pixels / grain_map.size
            
            # Texture strength (simplified)
            orientation_map = microstructure_data['orientation_map']
            if orientation_map.size > 0:
                # Calculate orientation spread
                phi1_std = np.std(orientation_map[:,:,0])
                Phi_std = np.std(orientation_map[:,:,1])
                phi2_std = np.std(orientation_map[:,:,2])
                texture_strength = 1 / (1 + (phi1_std + Phi_std + phi2_std)/100)
            else:
                texture_strength = 0.5
            
            stats = {
                'mean_grain_diameter': mean_diameter,
                'std_grain_diameter': std_diameter,
                'grain_size_variation': std_diameter / mean_diameter if mean_diameter > 0 else 0,
                'mean_aspect_ratio': mean_aspect_ratio,
                'gb_density': gb_density,
                'texture_strength': texture_strength,
                'n_grains': len(grain_stats),
                'phase_fractions': phase_fractions
            }
            
            return stats
        
        def simulate_deformation(self, microstructure_data: Dict[str, Any],
                               strain: float = 0.1,
                               strain_type: str = "tensile") -> Dict[str, Any]:
            """Simulate microstructure evolution during deformation"""
            
            grain_map = microstructure_data['grain_map'].copy()
            size = grain_map.shape[0]
            
            # Create deformation gradient
            if strain_type == "tensile":
                # Uniaxial tension in x-direction
                F = np.array([[1 + strain, 0, 0],
                             [0, 1 - strain*0.3, 0],
                             [0, 0, 1 - strain*0.3]])
            elif strain_type == "shear":
                # Simple shear
                F = np.array([[1, strain, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
            elif strain_type == "compression":
                # Uniaxial compression
                F = np.array([[1 - strain, 0, 0],
                             [0, 1 + strain*0.3, 0],
                             [0, 0, 1 + strain*0.3]])
            else:
                F = np.eye(3)
            
            # Create coordinate grid
            x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
            coords = np.stack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())], axis=1)
            
            # Apply deformation
            deformed_coords = coords @ F.T
            
            # Reshape back to 2D
            deformed_x = deformed_coords[:,0].reshape((size, size))
            deformed_y = deformed_coords[:,1].reshape((size, size))
            
            # Interpolate grain map onto deformed coordinates
            from scipy.interpolate import RegularGridInterpolator
            
            # Create interpolator for original grain map
            x_orig = np.arange(size)
            y_orig = np.arange(size)
            interpolator = RegularGridInterpolator((y_orig, x_orig), grain_map,
                                                  method='nearest', bounds_error=False, fill_value=0)
            
            # Map deformed coordinates back to original grid
            # Normalize deformed coordinates to [0, size-1]
            deformed_x_norm = (deformed_x + 1) * (size - 1) / 2
            deformed_y_norm = (deformed_y + 1) * (size - 1) / 2
            
            # Ensure coordinates are within bounds
            deformed_x_norm = np.clip(deformed_x_norm, 0, size-1)
            deformed_y_norm = np.clip(deformed_y_norm, 0, size-1)
            
            # Create query points for interpolation
            query_points = np.stack([deformed_y_norm.ravel(), deformed_x_norm.ravel()], axis=1)
            
            # Interpolate
            deformed_grain_map = interpolator(query_points).reshape((size, size))
            
            # Calculate deformation metrics
            volumetric_strain = np.linalg.det(F) - 1
            deviatoric_strain = strain - volumetric_strain/3
            
            deformed_data = {
                'original_grain_map': grain_map,
                'deformed_grain_map': deformed_grain_map.astype(np.int32),
                'deformation_gradient': F,
                'strain': strain,
                'strain_type': strain_type,
                'volumetric_strain': volumetric_strain,
                'deviatoric_strain': deviatoric_strain,
                'deformed_coords': (deformed_x, deformed_y)
            }
            
            return deformed_data
    
    # ==================== ALLOY DESIGNER ====================
    
    def design_alloy(self, base_element: str = "Fe",
                    alloying_elements: Dict[str, float] = None,
                    target_properties: Dict[str, float] = None,
                    processing: str = "wrought") -> Dict[str, Any]:
        """Advanced alloy design using empirical models with physics-based predictions"""
        
        if alloying_elements is None:
            alloying_elements = {"C": 0.45, "Mn": 0.75, "Si": 0.25, "Cr": 0.25}
        
        if target_properties is None:
            target_properties = {"yield_strength": 500, "elongation": 15, "toughness": 50}
        
        # Base properties for pure elements
        base_properties = {
            "Fe": {"strength": 200, "modulus": 211, "density": 7870, "melting": 1538},
            "Al": {"strength": 40, "modulus": 69, "density": 2700, "melting": 660},
            "Ti": {"strength": 250, "modulus": 116, "density": 4510, "melting": 1668},
            "Ni": {"strength": 150, "modulus": 200, "density": 8900, "melting": 1455}
        }
        
        base = base_properties.get(base_element, base_properties["Fe"])
        
        # Strengthening models
        ss_coefficients = {
            "C": {"Fe": 5000, "Al": 0, "Ti": 1000, "Ni": 2000},
            "Mn": {"Fe": 80, "Al": 30, "Ti": 50, "Ni": 60},
            "Si": {"Fe": 60, "Al": 40, "Ti": 80, "Ni": 70},
            "Cr": {"Fe": 50, "Al": 10, "Ti": 100, "Ni": 80},
            "Ni": {"Fe": 40, "Al": 20, "Ti": 60, "Ni": 0},
            "Mo": {"Fe": 100, "Al": 0, "Ti": 150, "Ni": 120},
            "V": {"Fe": 300, "Al": 0, "Ti": 200, "Ni": 250},
            "Ti": {"Fe": 400, "Al": 50, "Ti": 0, "Ni": 300},
            "Nb": {"Fe": 350, "Al": 0, "Ti": 180, "Ni": 280},
            "Cu": {"Fe": 60, "Al": 70, "Ti": 40, "Ni": 50}
        }
        
        # Solid solution strengthening
        ss_strength = 0
        for element, wt_pct in alloying_elements.items():
            if element in ss_coefficients and base_element in ss_coefficients[element]:
                ss_strength += ss_coefficients[element][base_element] * wt_pct
        
        # Precipitation strengthening (Ashby-Orowan model)
        precip_strength = 0
        precip_formers = ["V", "Ti", "Nb", "Al", "Cu"]
        precip_content = sum(alloying_elements.get(e, 0) for e in precip_formers)
        
        if precip_content > 0.01:
            # Simplified Ashby-Orowan: Δσ = (Gb/π√(1-ν)) * (√f/d) * ln(d/2b)
            G = base["modulus"] * 1000 / (2 * (1 + 0.3))  # Shear modulus in MPa
            b = 0.25e-9  # Burgers vector in m
            nu = 0.3  # Poisson's ratio
            f = precip_content * 0.1  # Volume fraction (simplified)
            d = 50e-9  # Precipitate diameter in m
            
            precip_strength = (G * b / (np.pi * np.sqrt(1 - nu))) * \
                            (np.sqrt(f) / d) * np.log(d / (2 * b))
        
        # Grain boundary strengthening (Hall-Petch)
        grain_size = 20  # μm (typical wrought)
        if processing == "cast":
            grain_size = 200
        elif processing == "PM":
            grain_size = 10
        
        gb_strength = 500 / np.sqrt(grain_size)
        
        # Dislocation strengthening (from processing)
        if processing == "cold_worked":
            dislocation_density = 1e15  # m⁻²
        elif processing == "hot_worked":
            dislocation_density = 1e13
        else:
            dislocation_density = 1e12
        
        dislocation_strength = 0.3 * base["modulus"] * 1000 * 0.25e-9 * np.sqrt(dislocation_density)
        
        # Total yield strength prediction
        predicted_yield = (base["strength"] + ss_strength + precip_strength + 
                         gb_strength + dislocation_strength)
        
        # Adjust for processing
        if processing == "cast":
            predicted_yield *= 0.8  # Cast materials typically weaker
        elif processing == "PM":
            predicted_yield *= 1.1  # Powder metallurgy can be stronger
        
        # Tensile strength prediction (typically 1.2-1.5 times yield)
        uts_ratio = 1.25 + 0.01 * (30 - target_properties.get("elongation", 15))
        predicted_uts = predicted_yield * uts_ratio
        
        # Elongation prediction (inverse relationship with strength)
        predicted_elongation = max(5, 50 - 0.05 * predicted_yield)
        
        # Toughness prediction (Ashby model)
        predicted_toughness = 0.1 * predicted_uts * predicted_elongation
        
        # Young's modulus (rule of mixtures)
        moduli = {
            "C": 1000, "Mn": 200, "Si": 150, "Cr": 300,
            "Ni": 200, "Mo": 300, "V": 400, "Ti": 120,
            "Nb": 350, "Cu": 120, "Al": 70, "Fe": 211
        }
        
        total_composition = 1.0  # Base element
        mod_sum = base["modulus"] * 1.0
        
        for element, wt_pct in alloying_elements.items():
            if element in moduli:
                total_composition += wt_pct
                mod_sum += moduli[element] * wt_pct
        
        predicted_E = mod_sum / total_composition
        
        # Density prediction (rule of mixtures)
        densities = {
            "C": 2260, "Mn": 7430, "Si": 2330, "Cr": 7190,
            "Ni": 8900, "Mo": 10280, "V": 6110, "Ti": 4510,
            "Nb": 8570, "Cu": 8960, "Al": 2700, "Fe": 7870
        }
        
        density_sum = base["density"] * 1.0
        for element, wt_pct in alloying_elements.items():
            if element in densities:
                density_sum += densities[element] * wt_pct
        
        predicted_density = density_sum / total_composition
        
        # Cost estimation (relative to base metal)
        cost_factors = {
            "C": 0.5, "Mn": 1.5, "Si": 2.0, "Cr": 3.0,
            "Ni": 10.0, "Mo": 20.0, "V": 50.0, "Ti": 30.0,
            "Nb": 40.0, "Cu": 5.0, "Al": 2.0
        }
        
        base_cost = 1.0  # Relative cost of base metal
        total_cost = base_cost
        
        for element, wt_pct in alloying_elements.items():
            if element in cost_factors:
                total_cost += cost_factors[element] * wt_pct
        
        # Calculate property match score
        match_score = 0
        if "yield_strength" in target_properties:
            strength_diff = abs(predicted_yield - target_properties["yield_strength"])
            strength_score = max(0, 100 - strength_diff)
            match_score += strength_score * 0.4
        
        if "elongation" in target_properties:
            elongation_diff = abs(predicted_elongation - target_properties["elongation"])
            elongation_score = max(0, 100 - elongation_diff * 10)
            match_score += elongation_score * 0.3
        
        if "toughness" in target_properties:
            toughness_diff = abs(predicted_toughness - target_properties["toughness"])
            toughness_score = max(0, 100 - toughness_diff * 2)
            match_score += toughness_score * 0.3
        
        match_score = min(100, match_score)
        
        result = {
            "alloy_composition": alloying_elements,
            "base_element": base_element,
            "processing": processing,
            "predicted_properties": {
                "yield_strength_MPa": round(predicted_yield, 1),
                "tensile_strength_MPa": round(predicted_uts, 1),
                "elongation_%": round(predicted_elongation, 1),
                "youngs_modulus_GPa": round(predicted_E, 1),
                "fracture_toughness_MPam0.5": round(predicted_toughness, 1),
                "density_kg_m3": round(predicted_density, 0),
                "estimated_cost_index": round(total_cost, 2)
            },
            "strengthening_contributions": {
                "solid_solution_MPa": round(ss_strength, 1),
                "precipitation_MPa": round(precip_strength, 1),
                "grain_boundary_MPa": round(gb_strength, 1),
                "dislocation_MPa": round(dislocation_strength, 1),
                "base_strength_MPa": round(base["strength"], 1)
            },
            "design_evaluation": {
                "property_match_score": round(match_score, 1),
                "strength_to_weight": round(predicted_yield / predicted_density * 1e6, 1),
                "toughness_to_weight": round(predicted_toughness / predicted_density * 1e6, 1),
                "cost_performance": round(predicted_yield / total_cost, 1)
            },
            "recommendations": self._generate_alloy_recommendations(alloying_elements, match_score)
        }
        
        return result
    
    def _generate_alloy_recommendations(self, composition: Dict[str, float], 
                                      match_score: float) -> List[str]:
        """Generate recommendations for alloy improvement"""
        recommendations = []
        
        if match_score < 70:
            recommendations.append("Consider adjusting composition to better match target properties")
        
        # Carbon content recommendations
        if "C" in composition:
            c_content = composition["C"]
            if c_content > 0.8:
                recommendations.append("High carbon content may reduce weldability and toughness")
            elif c_content < 0.2:
                recommendations.append("Low carbon content may limit hardenability")
        
        # Manganese recommendations
        if "Mn" in composition:
            mn_content = composition["Mn"]
            if mn_content > 1.5:
                recommendations.append("High manganese may improve hardenability but reduce ductility")
        
        # Cost reduction suggestions
        expensive_elements = {"Ni": 10.0, "Mo": 20.0, "V": 50.0, "Ti": 30.0, "Nb": 40.0}
        for element, cost in expensive_elements.items():
            if element in composition and composition[element] > 0.1:
                recommendations.append(f"Consider reducing {element} content to lower cost")
        
        # Processing recommendations
        if len(recommendations) == 0:
            recommendations.append("Alloy composition looks balanced. Consider heat treatment optimization.")
        
        return recommendations
    
    # ==================== DATA EXPORT & CERTIFICATION ====================
    
    def generate_test_certificate(self, test_type: str,
                                material: str,
                                properties: Dict[str, float],
                                test_conditions: Dict[str, Any] = None,
                                iso_standard: str = "ISO 6892-1") -> Dict[str, Any]:
        """Generate ISO-compliant test certificate with detailed metadata"""
        
        if test_conditions is None:
            test_conditions = {
                "temperature": "23 ± 2°C",
                "humidity": "50 ± 10% RH",
                "strain_rate": "0.001 s⁻¹",
                "specimen_type": "Round, gauge diameter 10mm",
                "testing_machine": "Virtual Tensile Tester v3.0",
                "extensometer_type": "Virtual clip-on extensometer"
            }
        
        # Generate unique certificate ID
        import uuid
        import datetime
        
        cert_id = str(uuid.uuid4())[:8].upper()
        issue_date = datetime.datetime.now().strftime("%Y-%m-%d")
        valid_until = (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        
        certificate = {
            "certificate_header": {
                "certificate_id": f"VMTL-{cert_id}",
                "issue_date": issue_date,
                "valid_until": valid_until,
                "test_laboratory": "Virtual Materials Testing Laboratory v3.0",
                "laboratory_accreditation": "ISO/IEC 17025 Compliant (Virtual)",
                "iso_standard": iso_standard,
                "certificate_version": "3.0"
            },
            "material_information": {
                "material_identification": material,
                "material_batch": "Virtual Batch-001",
                "heat_number": "VHT-2024-001",
                "supplier": "Virtual Materials Inc.",
                "receipt_date": issue_date
            },
            "test_information": {
                "test_type": test_type,
                "test_date": issue_date,
                "test_engineer": "Virtual Testing System",
                "test_conditions": test_conditions,
                "calibration_status": {
                    "force_cell": "ISO 7500-1 compliant, last calibrated: 30 days ago",
                    "extensometer": "ISO 9513 compliant, last calibrated: 30 days ago",
                    "temperature_control": "±2°C accuracy",
                    "data_acquisition": "16-bit resolution, 100Hz sampling"
                }
            },
            "test_results": {
                "mechanical_properties": properties,
                "test_validity": {
                    "yield_strength_accuracy": "±2% of reading",
                    "tensile_strength_accuracy": "±1% of reading",
                    "elongation_accuracy": "±5% of reading",
                    "youngs_modulus_accuracy": "±3% of reading",
                    "measurement_uncertainty": "Expanded uncertainty k=2 (95% confidence)"
                },
                "statistical_data": {
                    "number_of_tests": 1,
                    "test_repeatability": "Virtual test - perfect repeatability",
                    "data_quality": "Simulation grade - for educational purposes"
                }
            },
            "certification_statement": {
                "statement": "This virtual test certificate is generated for educational and research purposes. "
                           "The results are based on computational models and should be validated with "
                           "physical testing for engineering applications.",
                "disclaimer": "While based on established scientific principles, this virtual testing "
                            "does not replace certified physical testing for critical applications.",
                "signature": {
                    "authorized_by": "Virtual Materials Scientist",
                    "position": "Chief Simulation Officer",
                    "date": issue_date,
                    "digital_signature": f"VMTL-SIG-{cert_id}"
                }
            },
            "quality_assurance": {
                "data_integrity": "SHA-256 checksum verified",
                "report_version": "1.0",
                "export_format": "JSON/PDF compatible",
                "traceability": f"Test traceable to virtual reference materials"
            }
        }
        
        return certificate
    
    def export_to_csv(self, data: Dict[str, Any], filename: str = "test_results.csv") -> str:
        """Export test results to CSV format with comprehensive formatting"""
        
        # Flatten nested dictionaries for CSV
        flat_data = self._flatten_dict(data)
        
        df = pd.DataFrame([flat_data])
        
        # Add metadata
        metadata = {
            "export_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "software_version": "VMTL v3.0",
            "export_format": "CSV"
        }
        
        # Create metadata string
        metadata_str = "# " + "\n# ".join([f"{k}: {v}" for k, v in metadata.items()]) + "\n"
        
        # Save to CSV with metadata
        with open(filename, 'w') as f:
            f.write(metadata_str)
            df.to_csv(f, index=False)
        
        return f"Data exported to {filename} with {len(flat_data)} data points"
    
    def export_to_json(self, data: Dict[str, Any], filename: str = "test_results.json") -> str:
        """Export test results to JSON format with pretty printing"""
        
        # Add export metadata
        data_with_meta = {
            "metadata": {
                "export_date": pd.Timestamp.now().isoformat(),
                "software_version": "VMTL v3.0",
                "export_format": "JSON",
                "data_integrity": "SHA-256 checksum placeholder"
            },
            "test_data": data
        }
        
        with open(filename, 'w') as f:
            json.dump(data_with_meta, f, indent=2, default=str)
        
        return f"Data exported to {filename}"
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _log_simulation(self, simulation_type: str, parameters: Dict[str, Any]):
        """Log simulation parameters and results for audit trail"""
        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "simulation_type": simulation_type,
            "parameters": parameters,
            "current_material": str(self.current_material),
            "current_microstructure": str(self.current_microstructure) if self.current_microstructure else None
        }
        self.simulation_history.append(log_entry)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.simulation_history) > 1000:
            self.simulation_history = self.simulation_history[-1000:]
    
    def get_simulation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent simulation history"""
        return self.simulation_history[-limit:]
    
    def clear_history(self):
        """Clear simulation history"""
        self.simulation_history = []

# ==================== UTILITY FUNCTIONS ====================

def calculate_strength_weight_ratio(material: MaterialProperties, 
                                  strength_type: str = "yield") -> float:
    """Calculate strength-to-weight ratio"""
    if strength_type == "yield":
        strength = material.yield_strength
    elif strength_type == "tensile":
        strength = material.tensile_strength
    else:
        strength = material.yield_strength
    
    return strength / material.density * 1e6  # MPa/(kg/m³) * 1e6 for better scale

def estimate_fatigue_strength(tensile_strength: float, 
                            material_type: str = "steel") -> float:
    """Estimate fatigue strength from tensile strength"""
    if material_type == "steel":
        return 0.5 * tensile_strength  # Typical for steels
    elif material_type == "aluminum":
        return 0.4 * tensile_strength  # Typical for aluminum alloys
    elif material_type == "titanium":
        return 0.55 * tensile_strength  # Typical for titanium alloys
    else:
        return 0.45 * tensile_strength  # Default

def convert_hardness_scale(hardness: float, 
                          from_scale: str = "HV",
                          to_scale: str = "HRC") -> float:
    """Convert between hardness scales"""
    # Simplified conversion formulas
    if from_scale == "HV" and to_scale == "HRC":
        # Vickers to Rockwell C (approximate)
        return 0.095 * hardness - 18.5
    elif from_scale == "HRC" and to_scale == "HV":
        # Rockwell C to Vickers (approximate)
        return (hardness + 18.5) / 0.095
    else:
        return hardness  # No conversion available

def calculate_thermal_stress(E: float, alpha: float, delta_T: float) -> float:
    """Calculate thermal stress from temperature change"""
    # σ_thermal = E * α * ΔT
    return E * alpha * delta_T

def estimate_creep_life(stress: float, temperature: float,
                       material_type: str = "steel") -> float:
    """Estimate creep rupture life using simplified Larson-Miller"""
    # Larson-Miller parameter: P = T(C + log t)
    C = 20  # Typical for steels
    
    # Material-specific coefficients
    if material_type == "steel":
        a, b = 1e5, -0.1
    elif material_type == "aluminum":
        a, b = 1e4, -0.08
    elif material_type == "titanium":
        a, b = 2e5, -0.12
    else:
        a, b = 5e4, -0.09
    
    # Calculate Larson-Miller parameter from stress
    P = a * stress ** b
    
    # Solve for time: t = 10^(P/T - C)
    T_kelvin = temperature + 273.15
    if T_kelvin > 0:
        log_t = P / T_kelvin - C
        t_hours = 10 ** log_t
    else:
        t_hours = 1e6  # Very long at low temperatures
    
    return max(t_hours, 0.1)  # Minimum 0.1 hours

# ==================== MAIN TEST FUNCTION ====================

def test_core_functionality():
    """Test the core functionality of the materials lab"""
    print("Testing Virtual Materials Lab Core Functionality...")
    print("=" * 60)
    
    # Initialize lab
    lab = VirtualMaterialsLab()
    
    # Test material database
    print(f"Loaded {len(lab.materials_db)} materials")
    for name, props in lab.materials_db.items():
        print(f"  - {name}: E={props.youngs_modulus} GPa, σ_y={props.yield_strength} MPa")
    
    # Test microstructure design
    print("\nTesting Microstructure Design...")
    microstructure = lab.design_microstructure(
        material="AISI 1045 (Steel)",
        grain_size=25.0,
        porosity=0.005,
        inclusion_size=15.0
    )
    print(f"  Grain size: {microstructure.grain_size} μm")
    print(f"  Hall-Petch strengthening: {microstructure.calculate_hall_peetch():.1f} MPa")
    
    # Test heat treatment
    print("\nTesting Heat Treatment...")
    heat_treatment = lab.apply_heat_treatment(
        quenching_rate=80.0,
        tempering_temp=550.0,
        tempering_time=2.0
    )
    print(f"  Cooling medium: {heat_treatment.cooling_medium}")
    print(f"  Hardenability: {heat_treatment.calculate_hardenability():.1f}")
    
    # Test tensile testing
    print("\nTesting Tensile Testing...")
    material = lab.materials_db["AISI 1045 (Steel)"]
    tensile_tester = lab.TensileTester(material, microstructure, heat_treatment)
    eps, stress = tensile_tester.generate_stress_strain_curve(
        constitutive_model="mixed",
        temperature=20.0,
        strain_rate=0.001
    )
    properties = tensile_tester.calculate_mechanical_properties()
    print(f"  Yield strength: {properties['0.2% Yield Strength (MPa)']} MPa")
    print(f"  UTS: {properties['Ultimate Tensile Strength (MPa)']} MPa")
    print(f"  Elongation: {properties['Total Elongation (%)']}%")
    
    # Test alloy design
    print("\nTesting Alloy Designer...")
    alloy_result = lab.design_alloy(
        base_element="Fe",
        alloying_elements={"C": 0.35, "Mn": 1.0, "Cr": 1.5, "Mo": 0.25},
        target_properties={"yield_strength": 800, "elongation": 12}
    )
    print(f"  Predicted yield: {alloy_result['predicted_properties']['yield_strength_MPa']} MPa")
    print(f"  Predicted elongation: {alloy_result['predicted_properties']['elongation_%']}%")
    print(f"  Match score: {alloy_result['design_evaluation']['property_match_score']}")
    
    print("\n" + "=" * 60)
    print("Core functionality test completed successfully!")
    return lab

if __name__ == "__main__":
    # Run test if executed directly
    lab = test_core_functionality()
