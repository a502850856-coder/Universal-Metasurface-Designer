# Universal-Metasurface-Designer
Universal Metasurface Designer is a web-based interactive CAD platform specifically designed for engineering metasurfaces and Diffractive Optical Elements (DOEs).It bridges the gap between theoretical physics models and nanofabrication layouts (GDSII), with enhanced support for lithography constraints and real-world light source conditions .
✨ Key Features

1. Diverse Optical Modes (13+ Modes)

Built-in phase distribution models for various photonics applications:

Basic Focusing: Focusing Lens (Point), Cylindrical Lens, Off-axis Lens, Astigmatic Lens

Structured Light: Vortex Beam, Perfect Vortex, Bessel Beam (Axicon), Airy Beam

Higher-Order Modes: Laguerre-Gauss (LG), Hermite-Gauss (HG)

Arrays & Holography: Multi-focus, Optical Lattice, Gerchberg-Saxton (GS) Holography

2. Engineering-Ready Manufacturing Support

Automated GDSII Generation: One-click export of Python scripts that utilize gdspy / gdstk to generate industrial-grade .gds layout files.

Real Unit Library: Support for importing CSV data (Radius-Phase-Transmission) generated from FDTD sweeps, enabling "What You See Is What You Get" phase mapping.

3-bit Quantization: Simulate lithography constraints by discretizing continuous phase into 8 levels (0, π/4, ..., 7π/4).

Aperture Control: Intelligent circular aperture clipping.

3. Physics Simulation & Correction

Point Source Correction: Automatically compensates for spherical wavefront divergence, essential for fiber-tip integration or LED sources.

Real-time FFT Preview: Built-in lightweight scalar diffraction simulation to visualize focal plane intensity distribution in real-time.

Math Formula Visualization: Interactive panel displaying the analytical phase formula for the current mode.

🚀 Quick Start

Web Interface

This project is built with React + Tailwind CSS.

Clone the Repository

git clone [https://github.com/your-username/universal-metasurface-designer.git](https://github.com/your-username/universal-metasurface-designer.git)
cd universal-metasurface-designer


Install Dependencies

npm install
# or
yarn install


Start Development Server

npm start


Open http://localhost:3000 in your browser.

Backend / Export Scripts (Python)

The exported scripts (meta_design_xxx.py) require a Python environment:

Install Python Dependencies

pip install -r requirements.txt


Run Exported Script

python meta_design_perfect_vortex.py


This will generate a .gds layout file and a .csv data table in the current directory.

🛠️ Parameters

Parameter

Description

Typical Value

Wavelength

Operating wavelength

400 - 1550 nm

Period (P)

Unit cell period

200 - 800 nm

Grid Size (N)

Array resolution (NxN)

100 - 1000

Focal Length

Focal length

10 - 1000 μm

Use 3-bit

Enable 8-level phase quantization

True / False

Is Point Source

Enable point source wavefront compensation

True / False

🤝 Contributing
Issues and Pull Requests are welcome!
Especially if you have new phase formulas (e.g., Super-oscillatory Lens, Broadband Achromatic designs), please feel free to add them to the MODES list.

📜 License
This project is licensed under the MIT License.
Developed with ❤️ for the Nanophotonics Community.
