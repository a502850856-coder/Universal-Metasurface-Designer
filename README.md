<img width="1163" height="1100" alt="image" src="https://github.com/user-attachments/assets/86926fb8-1b09-459a-8c7c-e450ed33335b" /># Universal-Metasurface-Designer
Universal Metasurface Designer is a web-based interactive CAD platform specifically designed for engineering metasurfaces and Diffractive Optical Elements (DOEs).It bridges the gap between theoretical physics models and nanofabrication layouts (GDSII), with enhanced support for lithography constraints and real-world light source conditions .
<img width="1163" height="1100" alt="image" src="https://github.com/user-attachments/assets/3f7d1151-9f92-4d4e-af1b-820dc7ecef10" />

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
npm install or yarn install
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


这是一款基于 Web 的交互式 CAD 平台，专门用于超构表面 (Metasurfaces) 与衍射光学元件 (DOE) 的工程设计。
它打通了从理论物理模型到纳米加工版图 (GDSII) 之间的转化壁垒，并针对实际的光刻工艺限制与真实光源条件提供了深度支持。
✨ 核心特性丰富的光学模式 (13 种以上)系统内置了面向各类光子学应用的相位分布模型：
基础聚焦： 点聚焦透镜、柱面透镜、离轴透镜、像散透镜。
结构光： 涡旋光束、完美涡旋光束、贝塞尔光束（轴锥镜）、艾里光束。
高阶模式： 拉盖尔-高斯 (LG) 模式、厄米-高斯 (HG) 模式。
阵列与全息： 多焦点阵列、光晶格、基于 GS 算法 (Gerchberg-Saxton) 的全息图。
面向工程制造的全面支持自动生成 GDSII 版图： 一键导出 Python 脚本。该脚本通过调用 gdspy，可直接生成工业级的 .gds 版图文件。真实单元结构库： 支持导入 FDTD 参数扫描生成的 CSV 数据（半径-相位-透过率），真正实现“所见即所得”的相位映射。
3-bit 相位量化： 将连续的相位离散化为 8 个级次（0, π/4, ..., 7π/4），从而精确模拟光刻加工的实际限制。
孔径控制： 提供智能的圆形通光孔径裁剪功能。物理仿真与误差校正点光源校正： 自动补偿球面波的发散问题。这项功能在光纤端面集成和 LED 光源场景中必不可少。
实时 FFT 预览： 内置轻量级标量衍射计算，实时且直观地展示焦平面上的光强分布。
公式可视化： 配备交互式面板，直接展示当前所选模式的解析相位公式。
🚀 快速上手Web 端界面本项目采用 React + Tailwind CSS 开发。
克隆代码仓库Bashgit clone https://github.com/your-username/universal-metasurface-designer.git
cd universal-metasurface-designer
安装项目依赖Bashnpm install
或者使用 yarn install
启动开发服务器Bashnpm start
随后在浏览器中访问 http://localhost:3000 即可。后端与导出脚本 (Python)平台导出的脚本（例如 meta_design_xxx.py）需要在 Python 环境下运行：安装 Python 依赖库Bashpip install -r requirements.txt
执行导出脚本Bashpython meta_design_perfect_vortex.py
运行完毕后，当前目录下会自动生成 .gds 版图文件以及对应的 .csv 数据表。
🛠️ 关键参数说明参数名称含义说明典型设定值Wavelength
工作波长400 - 1550 nmPeriod (P)晶胞（单元结构）周期200 - 800 nm
Grid Size (N)阵列的网格分辨率 (N×N)100 - 1000
Focal Length透镜焦距10 - 1000 μmUse 3-bit是否开启 8 阶相位量化True / False
Is Point Source是否开启点光源波前补偿True / False🤝 参与贡献我们非常欢迎大家提交 Issue 和 Pull Request！
如果您掌握了新的相位公式（例如：超振荡透镜、宽带消色差设计），请务必将它们补充到 MODES 列表中。
📜本项目基于 MIT 协议 开源。
