# UK Rare Earth Elements (REE) Economic Impact Model

A coupled **Dynamic Input-Output (DIO) — Computable General Equilibrium (CGE) — Agent-Based Model (ABM)** framework for quantifying the macroeconomic and sectoral impacts of rare earth element supply disruptions on the United Kingdom economy.

---

## Overview

China controls ~90% of global REE refining. The May 2025 Chinese magnet export controls demonstrated that supply restrictions propagate rapidly through high-technology manufacturing supply chains. This model provides a multi-layered analytical framework to estimate:

- Sectoral output losses and GDP impacts under varying disruption severities
- Welfare effects (Equivalent Variation) on UK households
- Inventory depletion dynamics and firm-level substitution behaviour
- Supply-chain propagation via Leontief and Ghosh multipliers
- Multi-region spillovers via an MRIO block model

---

## Repository Structure

```
uk_ree_model/
├── app.py                    # Streamlit interactive dashboard
├── data/
│   └── uk_io_synthetic.py    # Synthetic UK IO table (12 sectors, ~2022 scale)
├── dio/
│   ├── leontief.py           # Static & Dynamic Leontief IO models
│   ├── ghosh.py              # Ghosh supply-side model
│   └── mrio.py               # Multi-Region IO (UK–China–RoW)
├── cge/
│   ├── sam_builder.py        # Social Accounting Matrix construction
│   ├── production.py         # Nested CES production functions
│   ├── trade.py              # Armington trade module
│   └── equilibrium.py        # CGE solver (Brent's method)
├── abm/
│   ├── agents.py             # Manufacturer, household, government agents
│   ├── market.py             # REE tâtonnement price mechanism
│   └── scheduler.py          # Mesa model scheduler
└── integration/
    ├── scenarios.py          # Five policy scenarios (A–E)
    └── coupling.py           # DIO–CGE–ABM coupling protocol
```

---

## Quick Start

```bash
pip install -r requirements.txt
cd uk_ree_model
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` with six tabs:
**Overview · DIO Analysis · CGE Analysis · ABM Simulation · Scenario Comparison · Sensitivity**

---

## Scenarios

| Label | Description | Peak θ | Duration |
|-------|-------------|--------|----------|
| A: Moderate | Partial export licensing, diplomatic exemption | 0.25 | 21 months |
| B: Severe | Replicates May 2025 magnet export collapse | 0.75 | 21 months |
| C: Sustained | Chronic disruption + substitution onset | 0.50 | 36 months |
| D: Complete | Systemic stress test | 1.00 | 24 months |
| E: Demand-side | Net Zero demand growth, no supply shock | 0.00 | 36 months |

---

## Mathematical Methodology

### 1. Input-Output Foundation

The UK economy is represented by a 12-sector IO table calibrated to approximate ONS 2022 proportions. The direct input coefficient matrix **A** satisfies:

$$A_{ij} = \frac{z_{ij}}{x_j}$$

where $z_{ij}$ is the monetary flow from sector $i$ to sector $j$ and $x_j$ is total output of sector $j$.

The **Leontief inverse** $\mathbf{L} = (\mathbf{I} - \mathbf{A})^{-1}$ satisfies the identity:

$$\mathbf{x} = \mathbf{L}\,\mathbf{y}$$

where $\mathbf{y} = (\mathbf{I} - \mathbf{A})\mathbf{x}$ is the final demand vector. Negative entries in $\mathbf{y}$ are economically valid for net-import sectors (e.g. REE, Steel) and are preserved to maintain the identity exactly.

#### Backward and Forward Linkages

**Backward linkage** (Rasmussen, 1956) — column sums of $\mathbf{L}$, measuring total output induced by one unit of final demand for sector $j$:

$$BL_j = \sum_i L_{ij}$$

**Forward linkage** — row sums of the Ghosh inverse $\mathbf{H} = (\mathbf{I} - \mathbf{G})^{-1}$, measuring how a sector's supply propagates downstream:

$$FL_i = \sum_j H_{ij}$$

where $G_{ij} = z_{ij}/x_i$ is the output allocation coefficient. This Ghosh-based measure is used in preference to row sums of $\mathbf{L}$, which understate the forward importance of supply-constrained sectors such as REE.

**Normalised linkages** are expressed relative to the sectoral mean:

$$\overline{BL}_j = \frac{BL_j}{\bar{BL}}, \qquad \overline{FL}_i = \frac{FL_i}{\bar{FL}}$$

Sectors with $\overline{BL} > 1$ and $\overline{FL} > 1$ are classified as **key sectors**.

#### Total REE Dependence

The total (direct + indirect) REE requirement per unit of final demand for sector $j$:

$$TRR_j = \sum_i r_i L_{ij}$$

where $\mathbf{r}$ is the REE intensity vector ($r_i$ = REE input per £ of output).

---

### 2. Ghosh Supply-Side Model

For a supply-side shock (Chinese export restriction), the Ghosh (1958) output-allocation model propagates the disruption downstream. The output vector satisfies:

$$\mathbf{x}' = \mathbf{v}' \mathbf{H}$$

where $\mathbf{v}$ is the primary input (value-added) vector. For net-import sectors where $v_k < 0$, a primary-input shock $v'_k = v_k(1-\theta)$ moves in the wrong direction. The correct formulation is an **output-based shock**:

$$\Delta v_k = -\frac{\theta \, x_k}{H_{kk}}$$

This guarantees $x'_k = (1-\theta)\,x_k$ regardless of the sign of $v_k$. The propagation to sector $j$ is:

$$\Delta x_j = \Delta v_k \cdot H_{kj} = -\frac{\theta\, x_k}{H_{kk}} \cdot H_{kj}$$

---

### 3. Multi-Region IO (MRIO)

The three-region (UK, China, RoW) block system is:

$$\begin{pmatrix} \mathbf{x}^U \\ \mathbf{x}^C \\ \mathbf{x}^R \end{pmatrix} = \begin{pmatrix} \mathbf{I} - \mathbf{A}^{UU} & -\mathbf{A}^{UC} & -\mathbf{A}^{UR} \\ -\mathbf{A}^{CU} & \mathbf{I} - \mathbf{A}^{CC} & -\mathbf{A}^{CR} \\ -\mathbf{A}^{RU} & -\mathbf{A}^{RC} & \mathbf{I} - \mathbf{A}^{RR} \end{pmatrix}^{-1} \begin{pmatrix} \mathbf{y}^U \\ \mathbf{y}^C \\ \mathbf{y}^R \end{pmatrix}$$

The Chinese export restriction shock reduces $\mathbf{A}^{CU}_{1,\cdot}$ (China's REE supply row to UK) by factor $(1-\theta)$.

---

### 4. Computable General Equilibrium (CGE)

#### Production Technology — Nested CES

Each sector $j$ uses a four-level nested CES production structure:

$$Y_j = \frac{1}{A_j} \cdot \text{CES}_{\sigma^{VA}}\!\left(\underbrace{\text{CES}_{\sigma^L}(L_j, K_j)}_{\text{value added}},\; \underbrace{\text{CES}_{\sigma^{INT}}\!\left(\underbrace{\text{CES}_{\sigma^A}(D_j^{REE}, M_j^{REE})}_{\text{Armington REE}},\; N_j\right)}_{\text{intermediates}}\right)$$

where $A_j$ is total factor productivity, $L_j, K_j$ are labour and capital, $D_j^{REE}, M_j^{REE}$ are domestic and imported REE, and $N_j$ is non-REE intermediates.

The **CES unit cost function** dual to the quantity index is:

$$c(\mathbf{p}) = \left[\alpha^\sigma p_1^{1-\sigma} + (1-\alpha)^\sigma p_2^{1-\sigma}\right]^{\frac{1}{1-\sigma}}$$

with the Cobb-Douglas limit ($\sigma \to 1$): $c = p_1^\alpha p_2^{1-\alpha}$.

**TFP calibration** ensures replication at baseline prices ($w_0 = r_0 = p^{REE}_0 = 1$):

$$A_j = c_j(w_0, r_0, p_0^{REE}) \implies c_j^*(w_0, r_0, p_0^{REE}) = 1 \quad \forall j$$

#### REE Import Price Schedule

The REE import price as a function of shock severity $\theta \in [0,1]$:

$$p^{REE}(\theta) = \begin{cases} 1 & \theta = 0 \\ \min\!\left(\dfrac{1}{(1-\theta)^{0.8}},\; 50\right) & \theta > 0 \end{cases}$$

This is calibrated to observed 2010 and 2023 REE price spikes: $p^{REE}(0.5) \approx 1.74\times$, $p^{REE}(0.75) \approx 3.03\times$.

#### Zero-Profit and Supply Response

Under perfect competition, output price equals unit cost:

$$p_j^* = c_j(w^*, r_0, p^{REE})$$

With sector-specific capital (short-run putty-clay), output responds to cost changes relative to the calibrated baseline ($c_j^0 = 1$):

$$Y_j^* = Y_j^0 \cdot \left(c_j^*\right)^{-\eta}, \qquad \eta = 1.5$$

clipped to $[10\%, 300\%]$ of baseline to prevent extreme realisations.

#### Labour Market Clearing

Labour is mobile across sectors. The equilibrium wage $w^*$ is the unique root of the labour market residual:

$$\mathcal{R}(w) = \frac{\displaystyle\sum_j \ell_j \, Y_j^*(w) - \bar{L}}{\bar{L}} = 0$$

where $\ell_j = \text{labour\_coeff}_j$ and $\bar{L} = \sum_j \ell_j Y_j^0$. This one-dimensional equation is solved by **Brent's method** on the bracket $w \in [0.05,\, 5.0]$.

#### GDP and Welfare

With sector-specific capital, GDP is measured by factor incomes:

$$\text{GDP}^* = \underbrace{w^* \bar{L}}_{\text{labour income}} + \underbrace{\sum_j k_j \, Y_j^*}_{\text{capital income (utilisation-weighted)}}$$

where $k_j$ is the capital coefficient for sector $j$.

**Consumer Price Index** (expenditure-share weighted):

$$\text{CPI} = \sum_j \mu_j^{\text{norm}} \, p_j^*, \qquad \mu_j^{\text{norm}} = \frac{y_j^{HH}}{\sum_k y_k^{HH}}$$

**Equivalent Variation** — real income change at base-year prices:

$$\text{EV} = \frac{I_H^{*\text{disp}}}{\text{CPI}} - I_H^{0\text{disp}}$$

where $I_H^{*\text{disp}} = (1 - s)(w^* \bar{L} + 0.3\, r_0 \bar{K} + G^{TR})$ and $s = 0.08$ is the household savings rate.

**Elasticity parameters** (short-run):

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| $\sigma^{VA}$ | 1.0 | Cobb-Douglas: VA vs intermediates |
| $\sigma^L$ | 0.5 | Labour–capital substitution |
| $\sigma^{INT}$ | 0.15–0.50 | REE–non-REE substitution (lower for REE-intensive) |
| $\sigma^A$ | 0.30 | Armington: domestic vs imported REE |
| $\eta$ | 1.5 | Supply elasticity |

---

### 5. Agent-Based Model (ABM)

#### Manufacturer Agents

Each manufacturer $m$ in sector $s$ follows an $(s, S)$ inventory policy with REE-criticality constraints. Monthly output is:

$$Q_m(t) = \min\!\left(\text{capacity}_m,\; \frac{\text{inv}_m(t)}{\text{ree\_intensity}_m}\right) \cdot \text{criticality}_m$$

where criticality $= \min(60 \cdot \text{ree\_intensity}_m,\; 1)$.

**Inventory dynamics:**

$$\text{inv}_m(t+1) = \text{inv}_m(t) + \text{purchases}_m(t) - \text{REE\_used}_m(t)$$

Reorder is triggered when $\text{inv}_m < s_m \cdot Q_m \cdot \text{ree\_intensity}_m$.

**Substitution** follows a logistic adoption curve with threshold $\bar{p}$:

$$\Pr(\text{substitute at } t) = \sigma\!\left(\lambda \cdot \frac{p^{REE}(t) - \bar{p}}{\bar{p}}\right)$$

Once substituted, effective REE intensity falls by 70%.

#### REE Market — Tâtonnement Pricing

The REE spot price adjusts to clear the market iteratively:

$$p^{REE}(t+1) = p^{REE}(t) \cdot \left(1 + \kappa \cdot \frac{D(t) - S(t)}{S(t)}\right)$$

where $\kappa = 0.3$ is the price adjustment speed, $D(t)$ is aggregate demand, and $S(t) = (1-\theta)\,x_1^0/12$ is monthly REE supply.

#### Government Stockpile Policy

The government releases stockpile when $p^{REE}(t) > p^{trigger}$:

$$\text{release}(t) = \min\!\left(\text{stockpile},\; \alpha_g \cdot D(t)\right)$$

where $\alpha_g = 0.15$ is the release fraction per period.

---

### 6. DIO–CGE–ABM Coupling Protocol

The three models exchange state variables each period $t$:

```
1.  DIO → ABM  :  sectoral output targets x*(t), IO coefficients A(t)
2.  CGE → ABM  :  equilibrium prices p*(t), factor incomes I*(t)
3.  ABM step   :  agents decide given p*(t), x*(t) → Q_m(t), inv_m(t)
4.  ABM → DIO  :  aggregate Q(t) updates final demand y(t+1)
5.  ABM → CGE  :  effective θ moderated by substitution progress
6.  CGE re-solves every 3 periods (cge_freq = 3)
```

**Consistency requirements:**

$$|p_j^{CGE} - p_j^{ABM}| / p_j^{CGE} < 5\% \quad \forall j$$
$$|\sum_m Q_m^{ABM} - x_j^{DIO}| / x_j^{DIO} < 10\% \quad \forall j$$

**Effective shock** after substitution:

$$\theta^{eff}(t) = \theta(t) \cdot \left(1 - 0.4 \cdot \bar{\phi}(t)\right)$$

where $\bar{\phi}(t)$ is the fraction of manufacturers that have substituted away from REE.

---

## Key Results

| Scenario | REE price peak | GDP loss | Welfare (EV) | Full substitution |
|----------|---------------|----------|--------------|-------------------|
| A (θ=0.25) | 2.67× | −0.98% | −£10.4bn | Month 20 |
| B (θ=0.75) | 3.26× | −5.50% | −£61.2bn | Month 15 |
| C (θ=0.50) | 2.92× | −2.51% | −£27.1bn | Month 30 |
| D (θ=1.00) | 3.51× | −26.6% | −£361bn | Month 20 |

Most exposed sectors at θ=0.75: REE −31.6%, Offshore Wind −27.9%, Electronics −26.9%, Automotive −26.8%, Aerospace −25.3%.

---

## Data Sources

| Data | Source |
|------|--------|
| IO table structure | ONS Supply & Use Tables (synthetic calibration) |
| REE intensity by sector | BGS CMIC 2024; JRC/IRENA engineering data |
| China import shares | BGS CMIC 2024; EU import data (proxy) |
| Sector output magnitudes | ONS UK National Accounts 2022 |
| Employment coefficients | ONS Labour Force Survey |
| Elasticity parameters | Koesler & Schymura (2015); Atalay (2017) |

---

## Citation

```bibtex
@software{uk_ree_model_2025,
  title  = {UK REE Economic Impact Model: A Coupled DIO-CGE-ABM Framework},
  author = {Zhang, Min},
  year   = {2025},
  url    = {https://github.com/zhangmin1006/uk-ree-model}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
