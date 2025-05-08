Update on 05/02/25:

- Shen Yu refined the train.py and encoder.py, the model should work now. Also add build_damaged.py to process the dataset with labels.

Update on 04/28/25:

- For data preprocessing and masking, etc., please see the data_preprocess branch, which Yumian Cui contributed. (5/4 + 5/5: Update: please see the preprocessing folder for preprocessing pipeline and the shared google drive for the processed data `Mural512_processed.zip`)
- The current Physics Refinement Module is all set to use!

Here is the wrokflow of DunHuang Mural Restoration process:
![alt text](https://github.com/rili0214/Dunhuang/blob/main/Images/efficient_workflow.png)

## 3.4 Physics Refinement Module

The Physics Refinement Module (PRM) serves as the analytical core of our restoration framework, integrating physical, chemical, and historical knowledge to characterize mural pigments with fidelity to their material properties.

### 3.4.1 System Architecture

The PRM integrates four core components in a sequential processing pipeline as illustrated in Fig. k {waiting for fig from other work}: (1) a spectral pigment database calibrated to Dunhuang materials, (2) a color-to-pigment mapper, (3) a thermodynamic validator, and (4) an optical properties simulator. The pipeline analyzes input image to identify possible historical pigments, validates mixture stability through physical modeling, and generates parameters for rendering and aging simulations.

### 3.4.2 Historical Pigment Database

The foundation of our approach is a comprehensive database of 35 pigments documented from the North Wei Dynasty to the Early Qing Dynasty in Dunhuang murals, partically derived from Tang Dynasty (Please see this reference paper: https://skytyz.dha.ac.cn/CN/Y2022/V1/I1/47 from Dunhuang Research Academy). Each pigment corresponds to a set of spectral reflectance curves (400-700nm), chemical formulae, microstructural characteristics (particle size distribution, porosity, surface roughness), and documented aging behavior derived from relevant studies (Please see this paper: https://www.sciencedirect.com/science/article/abs/pii/S0927024805000048 and this paper: https://arxiv.org/abs/physics/0505037). This database serves as both a reference for pigment identification and as training data for our mapper.

### 3.4.3 Color-to-Pigment Mapping

Our color-to-pigment mapping employs adaptive clustering to identify statistically significant color regions in the input image. Unlike previous approaches that rely solely on RGB values (please see https://www.sciencedirect.com/science/article/pii/S0039914023007087 and this: https://pmc.ncbi.nlm.nih.gov/articles/PMC9798325/), we implement a multi-feature analysis incorporating:

1. **Adaptive Color Clustering**: We dynamically determine optimal cluster counts ($k$) by analyzing color variance and spatial distribution in a dimensionally-reduced feature space. For regions with distinct boundaries, we employ K-means clustering; for gradients and subtle transitions, we switch to DBSCAN (Please see https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf) with parameters optimized for historical pigment properties.

2. **Spectral Feature Extraction**: From each cluster, we extract spectral features using the relationship:
   
   $$S(\lambda) = f_c(R, G, B, \lambda) + \sum_{i=1}^{n} w_i \cdot g_i(\lambda)$$
   
   where $S(\lambda)$ is the estimated spectral reflectance, $f_c$ is a color-to-spectrum mapping function, and $g_i$ are spectral basis functions weighted by $w_i$.

   Note: In the future work, we plan to replace this with the SWin transformer.

3. **Pigment Probability Mapping**: We calculate pigment probabilities through a joint embedding space that combines RGB, spectral, and microstructural features:
   
   $$P(p_i | \mathbf{x}) = \frac{\exp(f_{\theta}(\mathbf{x}, \mathbf{e_i}))}{\sum_{j=1}^{N} \exp(f_{\theta}(\mathbf{x}, \mathbf{e_j}))}$$
   
   where $\mathbf{x}$ represents input features, $\mathbf{e_i}$ is the embedding for pigment $i$, and $f_{\theta}$ is a learned compatibility function.

### 3.4.4 Thermodynamic Validation

We propose a thermodynamic validation model that assesses physical and chemical stability of identified pigment mixtures. This component addresses the critical gap in existing techniques that often ignore material compatibility.

The validator employs the Arrhenius equation (Please see https://en.wikipedia.org/wiki/Arrhenius_equation) for modeling reaction kinetics between pigment pairs:

$$k = A \exp\left(\frac{-E_a}{RT}\right)$$

where $k$ is the reaction rate constant, $A$ is the pre-exponential factor, $E_a$ is the activation energy, $R$ is the gas constant, and $T$ is temperature.

For each pigment pair $(p_i, p_j)$ in a mixture with concentration factors $(c_i, c_j)$, we calculate an instability contribution:

$$I(p_i, p_j) = c_i \cdot c_j \cdot k(p_i, p_j) \cdot s(p_i, p_j)$$

where $s(p_i, p_j)$ is a reaction severity factor determined by analyzing documented degradation patterns in historical samples.

The system models four classes of potential incompatibilities:
1. Chemical reactions (e.g., lead-sulfur interactions)
2. Nanoscale physical interactions affecting cohesion
3. Differential environmental responses
4. pH-dependent processes

The final stability score is computed as:

$$S = \max\left(0, \min\left(1, 1 - \sum_{i,j} I(p_i, p_j) - \phi(e) + \psi(h)\right)\right)$$

where $\phi(e)$ represents environmental penalties and $\psi(h)$ accounts for historically validated bonuses based on documented practices in the database.

### 3.4.5 Aging Simulation

Our aging simulation module simulates the changes in pigment properties through multi-stage degradation processes. For a given timespan $t$, degradation effects are modeled as:

$$E_d(t) = E_{max} \cdot (1 - e^{-r_d \cdot t \cdot f(e)})$$

where $E_d$ represents a specific effect (darkening, yellowing, fading, or cracking), $E_{max}$ is the maximum possible effect magnitude, $r_d$ is the effect-specific rate constant, and $f(e)$ is an environmental modulation function.

The system simulates spectral changes using:

$$S_{\text{aged}}(\lambda) = S_{\text{original}}(\lambda) \cdot M_d(\lambda) \cdot M_y(\lambda) \cdot M_f(\lambda)$$

where $M_d$, $M_y$, and $M_f$ are wavelength-dependent modulation functions for darkening, yellowing, and fading respectively. This approach enables prediction of color shifts with a less mean CIEDE2000 error compared to naturally aged samples.

### 3.4.6 Optical Properties Calculation

For physically-based rendering, we calculate optical properties using Kubelka-Munk theory for spectral mixing:

$$r_\infty = 1 + \frac{K}{S} - \sqrt{\left(\frac{K}{S}\right)^2 + 2\frac{K}{S}}$$

where $r_\infty$ is the reflectance of an infinitely thick layer, $K$ is the absorption coefficient, and $S$ is the scattering coefficient.

For mixtures of $n$ pigments with concentrations $c_i$, we compute:

$$\frac{K}{S}_{\text{mix}} = \sum_{i=1}^{n} c_i \cdot \frac{K}{S}_i$$

We further extend this with microstructural modeling of surface properties to generate BRDF parameters:

$$\omega_i = g_{\theta_i}(c, S(\lambda), \mu)$$

where $\omega_i$ represents a BRDF parameter (roughness, specular, etc.), $c$ is the vector of pigment concentrations, $S(\lambda)$ is the spectral reflectance, and $\mu$ contains microstructural parameters.

### 3.4.7 Integration and Validation

We introduce a bidirectional integration between physical modeling and historical knowledge. We validate generated pigment mixtures against period-appropriate constraints derived from art historical analysis.

The historical accuracy score for a mixture is calculated as:

$$A = \frac{|\{p_i \in P_m\} \cap \{p_j \in P_h\}|}{|P_m|}$$

where $P_m$ is the set of pigments in the mixture and $P_h$ is the set of historically documented pigments for the specific period and region.

This integration ensures that restoration recommendations balance physical stability with historical authenticity, providing conservators with solutions that are both materially sound and culturally accurate.
