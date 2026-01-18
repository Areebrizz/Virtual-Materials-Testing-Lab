🔬 Virtual Materials Testing Laboratory (VMTL)
==============================================

**Version 3.0 | ISO 6892-1 Aligned | Multi-scale Modeling Framework**

A comprehensive virtual laboratory for **materials science education and research**, featuring simulations of material testing, microstructure design, alloy development, and standards-aligned data analysis.

> This platform is intended for **education, research, and concept validation**. It does **not** replace certified physical testing laboratories.

🌟 Features
-----------

### Core Modules

#### ⚗️ Sample Preparation Station

*   Material selection from a curated ASM-based database
    
*   Microstructure design with Hall–Petch strengthening
    
*   Heat treatment simulation: quenching, tempering, aging
    

#### 📈 Tensile Testing Module

*   Stress–strain curve generation
    
*   Constitutive models: Hollomon, Voce
    
*   True stress–strain conversion and necking prediction
    
*   Interactive visualization using Plotly
    

#### 🔄 Fatigue Testing Module

*   S–N curve generation using the Basquin equation
    
*   Surface finish and stress ratio (R) effects
    
*   Fatigue life prediction
    

#### ⚡ Fracture Toughness Testing

*   Crack-tip stress field estimation
    
*   Plastic zone size calculation
    
*   Williams asymptotic expansion
    

#### 🔥 Creep Testing Module

*   Creep deformation using Norton’s law
    
*   Larson–Miller parameter evaluation
    
*   Rupture life prediction
    

#### 🔬 Microstructure Viewer

*   Voronoi tessellation-based microstructure generation
    
*   Interactive 2D/3D visualization
    
*   Grain size and statistical analysis
    

#### 🧪 Alloy Designer

*   Empirical alloy property prediction
    
*   Solid-solution and precipitation strengthening models
    
*   Custom alloy composition design
    

#### 📊 Data Export and Certification

*   Standards-aligned tensile test reports
    
*   CSV and JSON export
    
*   Quality documentation for coursework and research
    


📖 Usage Workflow
-----------------

1.  **Sample Preparation**
    
    *   Select a base material
        
    *   Define microstructural parameters
        
    *   Apply heat treatment conditions
        
2.  **Testing**
    
    *   Tensile testing
        
    *   Fatigue life estimation
        
    *   Fracture mechanics analysis
        
    *   Creep deformation simulation
        
3.  **Analysis**
    
    *   Interactive plots
        
    *   Mechanical property extraction
        
    *   Statistical uncertainty inspection
        
4.  **Export**
    
    *   Generate standards-aligned reports
        
    *   Export raw and processed data
        

🎓 Educational Applications
---------------------------

### Students

*   Virtual experiments without physical equipment
    
*   Interactive learning of structure–property relationships
    
*   Assignment-ready simulations with reproducible results
    

### Educators

*   Live lecture demonstrations
    
*   Virtual lab replacements
    
*   Quantitative assessment tools
    
*   Undergraduate research project support
    

🔬 Scientific Foundations
-------------------------

### Implemented Models

*   Hollomon and Voce constitutive laws
    
*   Basquin fatigue relation
    
*   Linear elastic fracture mechanics
    
*   Norton creep law and Larson–Miller parameter
    
*   Strengthening mechanisms: Hall–Petch, solid solution, precipitation
    

### Included Materials

*   AISI 1045 Steel
    
*   Aluminum 6061-T6
    
*   Ti-6Al-4V
    
*   Stainless Steel 316L
    

🧠 Software Architecture
------------------------

*   **Frontend:** Streamlit
    
*   **Visualization:** Plotly
    
*   **Numerical Computing:** NumPy, SciPy
    
*   **Data Handling:** Pandas
    

Performance is optimized for real-time interaction with classroom-scale workloads.

📐 Standards Alignment
----------------------

This project aligns conceptually with:

*   ISO 6892-1 (Tensile testing of metallic materials)
    
*   ISO 7500-1 (Force calibration)
    
*   ISO 9513 (Extensometer calibration)
    
*   ISO/IEC 17025 (Testing laboratory competence)
    

Alignment is **educational and methodological**, not certifying.

`

🤝 Contributing
---------------

Contributions are welcome via pull requests or issues. Please:

*   Follow PEP 8
    
*   Include docstrings and type hints
    
*   Add tests for new features
    

📄 License
----------

Licensed under the MIT License.

👨‍💻 Author
------------

**Muhammad Areeb Rizwan Siddiqui** Mechanical Engineer | Materials & Manufacturing Systems

*   Website: [https://www.areebrizwan.com](https://www.areebrizwan.com/)
    
*   LinkedIn: [https://www.linkedin.com/in/areebrizwan](https://www.linkedin.com/in/areebrizwan)
    

📈 Roadmap
----------

*   Expand material database
    
*   Add phase transformation modeling
    
*   Integrate FEM-based solvers
    
*   Introduce machine learning-assisted property prediction
    
*   Enable collaborative classroom features
    

_Advancing materials science education through virtual experimentation._
