# UK Rare Earth Elements Impact Analysis: Dynamic IO + CGE + Agent-Based Modelling
**Research Design & Implementation Plan | April 2026**

---

## 1. Research Architecture Overview

This study integrates three complementary methodologies into a unified simulation framework to analyse how REE supply shocks propagate through the UK economy:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT-BASED MODEL (ABM)                       │
│  Heterogeneous agents: firms, households, government, RoW        │
│  Adaptive behaviour, network topology, emergent dynamics         │
├─────────────────────────────────────────────────────────────────┤
│              COMPUTABLE GENERAL EQUILIBRIUM (CGE)                │
│  Price formation, factor markets, trade (Armington),             │
│  welfare effects, medium-run substitution                        │
├─────────────────────────────────────────────────────────────────┤
│              DYNAMIC INPUT–OUTPUT (DIO) MODEL                    │
│  Inter-industry linkages, Leontief/Ghosh propagation,            │
│  capital accumulation, REE intensity coefficients                │
└─────────────────────────────────────────────────────────────────┘
         ↑ calibration ↑            ↑ validation ↑
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  ONS IO tables · BGS · HMRC · BEIS · OECD ICIO · EXIOBASE       │
└─────────────────────────────────────────────────────────────────┘
```

### Integration Logic
- **DIO** provides the structural backbone: inter-industry coefficients, REE intensity, capital dynamics
- **CGE** adds price mechanisms and medium-run substitution that standard IO lacks
- **ABM** captures heterogeneous firm behaviour, adaptive expectations, and path-dependent dynamics that CGE equilibrium assumptions mask

---

## 2. Scope and Research Questions

### UK-Specific Focus
The UK presents a distinct case from the US/EU analysis in the existing REE_Supply_Chain_Risk_Analysis.md:
- Post-Brexit trade relationships alter REE import pathways (no Critical Raw Materials Act coverage)
- UK has specific sectoral exposures: offshore wind (North Sea), defence (BAE Systems, Rolls-Royce), automotive (Jaguar Land Rover, MINI), and electronics
- BGS manages a UK-specific Critical Minerals Intelligence Centre (CMIC)
- UK domestic REE deposits exist (Greenland-proximate claims, limited mainland deposits)

### Research Questions
1. What are the direct and indirect GDP losses to the UK under REE supply disruption scenarios (θ = 0.30, 0.50, 0.75)?
2. Which UK sectors and regions bear the highest adjustment costs?
3. How do UK firm-level adaptive behaviours (stockpiling, substitution, redesign) affect aggregate resilience?
4. What is the welfare impact (consumer prices, employment) under different policy responses?
5. How does the UK's post-Brexit position affect REE vulnerability relative to EU27?

---

## 3. Phase Plan

### Phase 1 — Data Assembly and IO Foundation (Months 1–3)
- Acquire and clean UK Supply & Use / IO tables from ONS
- Map REE flows to UK IO sectors using BGS CMIC data
- Construct UK REE intensity coefficient vector **r**
- Validate against OECD ICIO UK node

### Phase 2 — Dynamic IO Model Construction (Months 2–4)
- Build Leontief dynamic model with capital accumulation matrix **B**
- Implement Ghosh supply-side shock propagation
- Calibrate supply shock scenarios from existing REE_Supply_Chain_Risk_Analysis.md
- Run baseline and shock simulations; estimate output multipliers

### Phase 3 — CGE Model Construction (Months 3–6)
- Build UK multi-sector CGE with Armington trade
- Calibrate Social Accounting Matrix (SAM) from ONS data
- Implement CES production functions with REE as intermediate input
- Add factor markets (labour, capital) and government block

### Phase 4 — ABM Construction (Months 5–8)
- Define agent types, behavioural rules, and network topology
- Initialise agent state from DIO/CGE equilibrium
- Implement adaptive expectations and decision heuristics
- Run Monte Carlo experiments; extract emergent outcomes

### Phase 5 — Integration and Policy Scenarios (Months 7–10)
- Couple DIO ↔ CGE ↔ ABM with consistent state-variable handoff
- Run integrated scenarios: supply shocks + policy responses
- Sensitivity analysis and uncertainty quantification
- Validate against 2025 observed UK market data

---

## 4. Data Sources

### 4.1 UK Input–Output and National Accounts

| Source | Data | URL / Access |
|--------|------|--------------|
| **ONS Supply & Use Tables** | 105-sector UK IO, annual 1997–2022 | ons.gov.uk/economy/nationalaccounts/supplyandusetables |
| **ONS National Accounts (Blue Book)** | GDP, value added, factor income by sector | ons.gov.uk/economy/grossdomesticproductgdp/datasets/bluebook |
| **ONS Annual Business Survey** | Firm-level turnover, employment by sector | ons.gov.uk/businessindustryandtrade/business/businessservices/datasets/uknonfinancialbusinesseconomyukbusinessactivitysizeandlocation |
| **ONS Labour Market Statistics** | Employment, wages by occupation and industry | ons.gov.uk/employmentandlabourmarket |
| **HMRC UK Trade Info** | Import/export by commodity (HS codes 2846, 2805) | uktradeinfo.com |
| **ONS Regional Accounts** | GVA by region and sector (NUTS1/2) | ons.gov.uk/economy/grossvalueaddedgva |

### 4.2 REE-Specific UK Data

| Source | Data | URL / Access |
|--------|------|--------------|
| **BGS Critical Minerals Intelligence Centre (CMIC)** | UK critical mineral demand, supply risk scores, sector intensity | bgs.ac.uk/news/bgs-critical-minerals-intelligence-centre |
| **BGS World Mineral Statistics** | Global REE production by country, historical series | bgs.ac.uk/mineralsuk/statistics/worldStatistics.html |
| **BGS UK Minerals Yearbook** | UK consumption and import reliance by mineral | bgs.ac.uk/mineralsuk/statistics/ukStatistics.html |
| **BEIS Critical Minerals Strategy (2023)** | UK policy targets, sector dependencies | gov.uk/government/publications/uk-critical-mineral-strategy |
| **BEIS/DESNZ Industrial Decarbonisation Strategy** | Sectoral roadmaps affecting REE demand | gov.uk/government/publications/industrial-decarbonisation-strategy |

### 4.3 International IO and Trade Data

| Source | Data | URL / Access |
|--------|------|--------------|
| **OECD ICIO 2025 Edition** | 76 countries, 45 sectors, GVC linkages | oecd.org/sti/ind/inter-country-input-output.htm |
| **EXIOBASE 3.8** | 44 countries + RoW, 200 products, material-flow extensions | exiobase.eu |
| **WIOD 2016 Release** | 43 countries, 56 sectors, time series | wiod.org |
| **UN Comtrade** | HS 280530, 2846, 854800 (magnets) bilateral trade flows | comtradeplus.un.org |
| **Eurostat PRODCOM** | EU manufacturing output of REE-intensive products | ec.europa.eu/eurostat/web/prodcom |

### 4.4 REE Price and Market Data

| Source | Data | URL / Access |
|--------|------|--------------|
| **USGS Mineral Commodity Summaries 2025–2026** | Global production, prices, trade | pubs.usgs.gov/periodicals/mcs2026 |
| **Metal Pages / Fastmarkets** | Daily/weekly REE spot prices (NdPr oxide, Dy oxide, etc.) | fastmarkets.com (subscription) |
| **Shanghai Metals Market (SMM)** | Chinese domestic and export prices | metal.com |
| **DERA (German Mineral Resources Agency)** | REE price indices, criticality scores | deutsche-rohstoffagentur.de |

### 4.5 Sector-Specific UK Data

| Source | Sector | Data |
|--------|--------|------|
| **SMMT (Society of Motor Manufacturers)** | Automotive | UK EV production, REE per vehicle |
| **RenewableUK / BEIS DESNZ** | Offshore wind | Installed capacity, magnet intensity per MW |
| **BAE Systems / Rolls-Royce Annual Reports** | Defence | REE-dependent product lines |
| **ADS Group** | Aerospace & Defence | Sector REE dependency |
| **UK Finance / Bank of England** | Financial sector | Supply-chain credit risk data |

### 4.6 Social Accounting Matrix (SAM) for CGE Calibration

| Source | Data |
|--------|------|
| **ONS Supply & Use Tables** | Primary SAM structure |
| **ONS Sector & Financial Accounts** | Capital and financial flows |
| **HMRC Corporation Tax Statistics** | Profit rates by sector |
| **ONS Household Finance Survey** | Household income and spending |
| **OBR Economic Fiscal Outlook** | Government budget constraint |

---

## 5. Mathematical Models

### 5.1 Dynamic Input–Output (DIO) Model

#### 5.1.1 Static Leontief Foundation

$$\mathbf{x} = (\mathbf{I} - \mathbf{A})^{-1} \mathbf{y}$$

Where:
- $\mathbf{x} \in \mathbb{R}^n$ = sectoral output vector ($n$ = number of UK IO sectors, ~105 ONS sectors or 45 ICIO sectors)
- $\mathbf{A} \in \mathbb{R}^{n \times n}$ = direct input coefficient matrix: $a_{ij} = z_{ij}/x_j$
- $\mathbf{y} \in \mathbb{R}^n$ = final demand vector (household, government, investment, exports)
- $\mathbf{L} = (\mathbf{I} - \mathbf{A})^{-1}$ = Leontief inverse

#### 5.1.2 REE Intensity Coefficients

Direct REE requirement vector:
$$\mathbf{r} = \mathbf{e}_{REE} \cdot (\mathbf{I} - \mathbf{A})^{-1}$$

Where $\mathbf{e}_{REE}$ is the row vector of direct REE inputs per unit output for each sector (constructed from BGS CMIC and engineering literature).

Total (direct + indirect) REE requirement:
$$\mathbf{TRR} = \mathbf{r} \cdot \mathbf{L}$$

This captures hidden upstream REE dependencies — e.g., an offshore wind manufacturer sourcing steel that requires REE-based catalysts in its production.

#### 5.1.3 Dynamic Extension: Capital Accumulation

Following Leontief's (1970) dynamic IO model:

$$(\mathbf{I} - \mathbf{A})\mathbf{x}(t) - \mathbf{B}[\mathbf{x}(t+1) - \mathbf{x}(t)] = \mathbf{c}(t)$$

Where:
- $\mathbf{B} \in \mathbb{R}^{n \times n}$ = capital coefficients matrix: $b_{ij}$ = capital goods of type $i$ required per unit increase in output capacity of sector $j$
- $\mathbf{c}(t)$ = non-investment final demand at time $t$
- $\mathbf{x}(t+1) - \mathbf{x}(t)$ = capacity expansion vector

Rearranging as a difference equation:

$$\mathbf{x}(t+1) = \mathbf{B}^{-1}[(\mathbf{I} - \mathbf{A})\mathbf{x}(t) - \mathbf{c}(t)] + \mathbf{x}(t)$$

This governs the time-path of sectoral outputs under REE shocks with capital adjustment.

#### 5.1.4 Supply-Side Ghosh Propagation

For supply-constrained REE shocks, we use the Ghosh (1958) output inverse:

$$\mathbf{x}' = \mathbf{v}' (\mathbf{I} - \mathbf{G})^{-1}$$

Where:
- $\mathbf{G} \in \mathbb{R}^{n \times n}$ = output allocation coefficient matrix: $g_{ij} = z_{ij}/x_i$
- $\mathbf{v}$ = primary input (value added) vector
- $(\mathbf{I} - \mathbf{G})^{-1}$ = Ghosh inverse

**Supply shock implementation:**

$$\Delta x_k = -\theta \cdot x_k^0 \quad \forall k \in \mathcal{K}_{REE}$$

Where $\mathcal{K}_{REE}$ = set of REE-supplying sectors, $\theta \in \{0.30, 0.50, 0.75\}$ as per existing scenario design.

Propagated output loss:
$$\Delta \mathbf{x} = (\mathbf{I} - \mathbf{G})^{-1} \Delta \mathbf{v}_{REE}$$

#### 5.1.5 UK-Specific Hybrid Model

Because the UK is a net REE importer, we use a **multi-regional IO (MRIO)** extension:

$$\begin{pmatrix} \mathbf{x}^{UK} \\ \mathbf{x}^{CN} \\ \mathbf{x}^{RoW} \end{pmatrix} = \left(\mathbf{I} - \begin{pmatrix} \mathbf{A}^{UU} & \mathbf{A}^{UC} & \mathbf{A}^{UR} \\ \mathbf{A}^{CU} & \mathbf{A}^{CC} & \mathbf{A}^{CR} \\ \mathbf{A}^{RU} & \mathbf{A}^{RC} & \mathbf{A}^{RR} \end{pmatrix}\right)^{-1} \begin{pmatrix} \mathbf{y}^{U} \\ \mathbf{y}^{C} \\ \mathbf{y}^{R} \end{pmatrix}$$

The Chinese export restriction modifies $\mathbf{A}^{UC}$ (UK's intermediate imports from China) directly.

---

### 5.2 Computable General Equilibrium (CGE) Model

#### 5.2.1 Production Structure

Each sector $j$ uses a nested CES production function:

**Top level** — Value added (VA) vs. intermediates composite (INT):

$$Y_j = \text{CES}(VA_j,\ INT_j;\ \sigma^{VA})$$

$$Y_j = A_j \left[\alpha_j VA_j^{\frac{\sigma_j - 1}{\sigma_j}} + (1-\alpha_j) INT_j^{\frac{\sigma_j - 1}{\sigma_j}}\right]^{\frac{\sigma_j}{\sigma_j - 1}}$$

**Intermediate nest** — REE composite vs. other intermediates:

$$INT_j = \text{CES}(REE_j,\ NREE_j;\ \sigma^{INT})$$

Where $\sigma^{INT}$ = substitution elasticity between REE inputs and other intermediates (low, ~0.1–0.3 based on engineering constraints).

**Value added nest** — Labour vs. Capital:

$$VA_j = \text{CES}(L_j,\ K_j;\ \sigma^{VA})$$

#### 5.2.2 REE as Critical Intermediate

REE inputs to sector $j$ combine domestic supply and imports via Armington aggregation:

$$REE_j = \left[\delta_j (REE_j^{dom})^{\frac{\sigma^A - 1}{\sigma^A}} + (1-\delta_j)(REE_j^{imp})^{\frac{\sigma^A - 1}{\sigma^A}}\right]^{\frac{\sigma^A}{\sigma^A - 1}}$$

Where $\sigma^A$ = Armington elasticity for REE (low, ~0.3, reflecting limited substitutability of Chinese vs. non-Chinese REE sources in the short run).

#### 5.2.3 Household Optimisation

Representative household maximises utility subject to budget constraint:

$$\max_{C_1,...,C_n} U = \prod_{i=1}^n C_i^{\mu_i}, \quad \text{s.t.} \quad \sum_{i=1}^n p_i C_i = I_H$$

Where $\mu_i$ = expenditure shares (calibrated from ONS Living Costs and Food Survey), $I_H$ = household income from labour, capital, and transfers.

#### 5.2.4 Government Block

$$G = T^{VAT} + T^{corp} + T^{inc} + T^{tariff} - TR - Sub$$

Government budget constraint:
$$\sum_i g_i p_i = \bar{G} \quad \text{(fixed real spending)}$$

**Policy instrument:** Import tariffs on REE ($t^{REE}$) and/or subsidies to domestic REE processing ($s^{dom}$).

#### 5.2.5 Trade and Armington

UK exports and imports via Armington (1969) assumption — domestic and foreign goods are imperfect substitutes:

**Import demand:**
$$M_i = \left(\frac{p_i^d}{p_i^m \cdot (1+t_i)}\right)^{\sigma_i^A} \cdot D_i^{Arm}$$

**Export supply:**
$$E_i = \left(\frac{p_i^x}{p_i^d}\right)^{\eta_i} \cdot Y_i^{exp}$$

Trade balance:
$$\sum_i p_i^x E_i = \sum_i p_i^m M_i + \bar{CAB}$$

Where $\bar{CAB}$ = fixed current account balance (closure rule).

#### 5.2.6 Market Clearing Conditions

**Goods markets:**
$$Y_j = C_j + G_j + I_j + E_j + \sum_k z_{jk} \quad \forall j$$

**Labour market:**
$$\sum_j L_j = \bar{L} \quad \text{(fixed labour supply)} \quad \text{or} \quad w = f(u, \phi) \quad \text{(wage curve)}$$

**Capital market:**
$$\sum_j K_j = \bar{K} \quad \text{(short run: sector-specific capital)}$$
$$r_j = r_j^{ss} + \lambda(K_j - K_j^*) \quad \text{(medium run: capital mobility)}$$

**REE market:**
$$REE^{supply}(\theta) = (1-\theta) \cdot REE^{supply,0}$$

Price adjustment:
$$p_{REE}(t) = p_{REE}^0 \cdot \left(\frac{REE^{demand}}{REE^{supply}(\theta)}\right)^{1/\epsilon}$$

Where $\epsilon$ = price elasticity of REE supply (calibrated from 2025 price spike data).

#### 5.2.7 Welfare Measure: Equivalent Variation

$$EV = I_H^0 \cdot \left[\frac{U^1}{U^0} - 1\right]$$

Or using the expenditure function:
$$EV = e(\mathbf{p}^0, U^1) - e(\mathbf{p}^0, U^0)$$

This captures the welfare loss from REE supply disruption in £ equivalent terms.

---

### 5.3 Agent-Based Model (ABM)

#### 5.3.1 Agent Taxonomy

| Agent Type | Count | Key Attributes |
|------------|-------|----------------|
| **REE Supplier agents** | $N_S$ (5–20) | country of origin, export quota $\theta$, price markup, contract structure |
| **Manufacturer agents** | $N_M$ (100–500) | sector, REE intensity, inventory, substitution capacity, adaptive expectations |
| **Household agents** | $N_H$ (representative groups) | income class, consumption bundle, labour supply |
| **Government agent** | 1 | tax policy, stockpile policy, subsidy instruments |
| **Financial agent** | 1–5 | credit supply, interest rate, supply-chain finance |
| **Foreign (RoW) agents** | $N_F$ | EU, US, RoW blocs; competing for REE supply |

#### 5.3.2 Manufacturer Agent Decision Rules

Each manufacturer agent $m$ in sector $j$ at time $t$:

**Production decision:**
$$Q_m(t) = \min\left(\bar{Q}_m(t),\ \frac{REE_m^{avail}(t)}{r_j}\right)$$

Where $\bar{Q}_m(t)$ = capacity-constrained output, $r_j$ = REE intensity of sector $j$.

**Inventory rule (s,S policy):**
$$\text{Order}_{m}(t) = \begin{cases} S_m - INV_m(t) & \text{if } INV_m(t) \leq s_m \\ 0 & \text{otherwise} \end{cases}$$

Where $s_m$ = reorder point, $S_m$ = target inventory level (heterogeneous across agents, drawn from distribution $\mathcal{N}(\bar{s}, \sigma_s^2)$).

**Substitution decision (adaptive):**

$$P(\text{substitute at }t) = \Phi\left(\frac{p_{REE}(t) - p_{REE}^{sub}}{\sigma_{sub}}\right)$$

Where $p_{REE}^{sub}$ = threshold price for switching to substitute technology (agent-specific), $\Phi$ = standard normal CDF.

**Expectations formation:**
$$p_{REE}^e(t+1) = \lambda_m p_{REE}(t) + (1-\lambda_m) p_{REE}^e(t)$$

Where $\lambda_m \in [0,1]$ = agent-specific learning rate (heterogeneous; some agents more forward-looking).

#### 5.3.3 Supplier Agent Behaviour

Chinese REE exporter agent:

$$x_k^{CN}(t) = (1-\theta(t)) \cdot x_k^{CN,0} \cdot \exp(\epsilon_t)$$

Where $\epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$ = idiosyncratic production noise.

Export licensing allocation rule:
$$\text{Allocation}_{c}(t) = \frac{share_c \cdot x_k^{CN}(t)}{\sum_{c'} share_{c'}}$$

Where allocation is pro-rated by historical trade share (UK receives $share_{UK} \approx 0.02–0.04$ of Chinese exports).

#### 5.3.4 Network Topology: Supply Chain Graph

Define a directed supply chain graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- Nodes $\mathcal{V}$ = all agents (suppliers, processors, manufacturers, households)
- Edges $\mathcal{E}$ = supply relationships with edge weights $w_{ij}$ = transaction volume

Propagation rule: if agent $i$ experiences supply failure:
$$\text{Risk}_{j}(t) = 1 - \prod_{i \in \mathcal{N}(j)} (1 - \rho_{ij} \cdot \mathbf{1}[\text{fail}_i(t)])$$

Where $\mathcal{N}(j)$ = upstream suppliers of $j$, $\rho_{ij}$ = dependency weight.

Network resilience metric:
$$\mathcal{R}(\mathcal{G}) = 1 - \frac{|\mathcal{V}_{fail}|}{|\mathcal{V}|}$$

#### 5.3.5 ABM Calibration from DIO/CGE

The ABM is initialised from the DIO/CGE equilibrium:
- Agent output targets: $Q_m^0 = x_j^{DIO} / N_M^j$ (share of IO sector output)
- REE flows: $REE_m^0 = r_j \cdot Q_m^0$
- Prices: $p^0$ from CGE equilibrium
- Inventory: $INV_m^0$ drawn from log-normal distribution calibrated to ONS business survey data

#### 5.3.6 Macroeconomic Aggregation

From individual agent decisions to aggregate outcomes:

$$GDP(t) = \sum_m Q_m(t) \cdot p_m(t) - \sum_m \text{Inputs}_m(t) \cdot \mathbf{p}_{inputs}(t)$$

$$\Delta GDP(t) = GDP(t) - GDP^0$$

$$\text{Employment}(t) = \sum_m L_m(t) = \sum_m \frac{Q_m(t)}{A_m(t)} \cdot \ell_j$$

Where $\ell_j$ = labour intensity of sector $j$.

---

### 5.4 DIO–CGE–ABM Coupling Protocol

The three models exchange information at each simulation period $t$:

```
Period t:
  1. DIO → ABM: provides sectoral output targets x*(t), IO coefficients A(t)
  2. CGE → ABM: provides equilibrium prices p*(t), factor incomes I*(t)
  3. ABM executes: agents make decisions given p*(t), x*(t); generate Q_m(t), INV_m(t)
  4. ABM → DIO: aggregate Q(t) updates final demand y(t+1) via investment decisions
  5. ABM → CGE: aggregate demand shifts update excess demand functions; CGE re-solves
  6. CGE → DIO: updated price vector feeds into IO valuation; A matrix updated if needed
  7. Loop to t+1
```

**Consistency requirements:**
- Nominal values consistent across models: $p^{CGE}(t) = p^{ABM}(t)$
- Physical flows consistent: $\sum_m Q_m^{ABM}(t) = x_j^{DIO}(t) \pm \epsilon$ (within tolerance)
- Trade balance: $CAB^{CGE}(t) = CAB^{DIO}(t)$

---

## 6. Shock Scenarios

Building on the existing REE_Supply_Chain_Risk_Analysis.md scenarios, calibrated for UK context:

| Scenario | θ | Duration | UK-Specific Features |
|----------|---|----------|----------------------|
| **A: Moderate** | 0.30 | 12 months | UK receives partial licensing exemption via diplomatic channels; £ depreciation offsets some cost increase |
| **B: Severe** | 0.75 | 6 months | Replicates May 2025 magnet export collapse; UK faces same EU-level supply denial |
| **C: Sustained** | 0.50 | 24 months | Chronic disruption; UK firms begin substitution and redesign after 6 months |
| **D: Complete** | 1.00 | 3 months | Worst-case; tests systemic resilience; triggers emergency stockpile release |
| **E: Demand-side** | 0.00 | 36 months | No shock, but UK Net Zero demand growth (EVs + offshore wind) raises REE requirements by 40% |

---

## 7. UK Sector Impact Matrix

| UK Sector | ONS Code | REE Intensity | Primary REEs | ABM Agent Count |
|-----------|----------|---------------|--------------|-----------------|
| Automotive (EV) | SIC 29 | High | Nd, Pr, Dy, Tb | 15–30 firms |
| Aerospace & Defence | SIC 30 | High | Dy, Tb, Sm, Y | 10–20 firms |
| Offshore Wind | SIC 3511 | High | Nd, Pr, Dy | 8–15 developers |
| Electronics | SIC 26 | Moderate–High | Nd, Eu, Y, Ce | 30–50 firms |
| Oil & Gas Refining | SIC 192 | Moderate | La, Ce | 5–10 refineries |
| Medical Devices | SIC 2660 | Moderate | Y, Gd | 15–25 firms |
| Industrial Machinery | SIC 28 | Low–Moderate | Nd, Ce | 20–40 firms |
| Steel & Metals | SIC 24 | Low | Ce, La | 10–15 firms |

---

## 8. Key Performance Indicators

### Economic Indicators
- **GDP impact** (£bn, % change from baseline)
- **GVA by sector** (value added lost per sector)
- **Employment** (FTE jobs affected)
- **Consumer Price Index** (REE-exposed component)
- **Equivalent Variation** (welfare loss, £bn)
- **Trade balance** (current account impact)

### Supply Chain Indicators
- **Supply chain disruption index**: $SCDI = \sum_j TRR_j \cdot \Delta x_j / \sum_j x_j$
- **Network resilience**: $\mathcal{R}(\mathcal{G})$
- **Mean inventory depletion time** across manufacturer agents
- **Fraction of agents triggering substitution** at each scenario

### Regional Indicators
- GVA impact by NUTS1 region (particularly Scotland for offshore wind, North East for Nissan Sunderland, South West for defence)
- Employment impact by LEP (Local Enterprise Partnership) area

---

## 9. Software Implementation

### Recommended Stack

| Component | Tool | Rationale |
|-----------|------|-----------|
| DIO model | Python (NumPy/SciPy) | Matrix operations, time-series simulation |
| CGE model | GAMS or Python (PuLP/CVXPY) | Nonlinear equation solving; GAMS has established CGE libraries (e.g., GTAP, ORANI-style) |
| ABM | Mesa (Python) or NetLogo | Mesa preferred for Python integration and reproducibility |
| Data pipeline | pandas, ONS API | UK data ingestion and cleaning |
| Visualisation | matplotlib, plotly, geopandas | Economic charts + regional maps |
| Monte Carlo | NumPy, joblib | Parallel runs for uncertainty quantification |

### Model File Structure
```
uk_ree_model/
├── data/
│   ├── ons_io_tables/          # ONS Supply & Use tables
│   ├── bgs_cmic/               # BGS REE intensity data
│   ├── hmrc_trade/             # HMRC HS 2846, 280530
│   └── oecd_icio/              # OECD ICIO UK slice
├── dio/
│   ├── leontief.py             # Static and dynamic Leontief
│   ├── ghosh.py                # Supply-side shock propagation
│   └── mrio.py                 # Multi-region IO extension
├── cge/
│   ├── sam_builder.py          # Social Accounting Matrix
│   ├── production.py           # CES production functions
│   ├── trade.py                # Armington trade module
│   └── equilibrium.py          # Market clearing solver
├── abm/
│   ├── agents.py               # Manufacturer, supplier, household agents
│   ├── network.py              # Supply chain graph
│   ├── scheduler.py            # ABM time loop
│   └── metrics.py              # Aggregation and KPIs
├── integration/
│   ├── coupling.py             # DIO–CGE–ABM handoff protocol
│   └── scenarios.py            # Scenario definitions
└── analysis/
    ├── sensitivity.py          # Parameter sensitivity
    └── visualisation.py        # Charts and maps
```

---

## 10. Validation Strategy

### 10.1 Historical Validation
- Backcast DIO model against ONS IO tables 2018–2022 (pre-shock baseline)
- Validate CGE trade elasticities against observed UK import responses
- Calibrate ABM inventory rules against ONS business survey data

### 10.2 Real-Time Validation Against 2025 Shock
- Compare simulated Scenario B (θ = 0.75) UK sectoral output impacts against:
  - HMRC trade data: UK REE import volumes H1 2025
  - ONS Index of Production: automotive and electronics output
  - CPI sub-indices for electronics and energy equipment
  - UK Wind Energy Database: any turbine delivery delays
- Use observed 2025 price spikes (Dy +168%, Y +598%) to validate CGE price formation

### 10.3 Cross-Model Consistency Checks
- DIO output multipliers within ±5% of OECD ICIO UK-slice multipliers
- CGE equilibrium prices reproduce ONS deflators for REE-intensive sectors
- ABM aggregate output converges to DIO solution at steady state

---

## 11. Connection to Existing Analysis

This plan extends REE_Supply_Chain_Risk_Analysis.md in the following ways:

| Existing Analysis | Extension in This Plan |
|-------------------|------------------------|
| US/EU focus on IO model | UK-specific ONS IO tables + BGS data |
| Static Leontief and Ghosh | Dynamic IO with capital matrix **B** |
| Supply shock scenarios θ ∈ {0.30, 0.75} | Same scenarios + UK diplomatic adjustment + demand-side scenario E |
| HHI concentration analysis | CGE price formation calibrated to observed 2025 spikes |
| ECB network finding (3 intermediaries) | ABM supply chain graph replicates and extends with firm-level dynamics |
| USGS nonlinear optimisation | CGE with CES substitution for medium-run adjustment |
| No stockpiling/inventory modelling | ABM (s,S) inventory rules capture stockpile depletion dynamics |
| No agent heterogeneity | ABM with heterogeneous firms, adaptive expectations, learning rates |

---

## 12. Key References for UK Context

- ONS (2024). *UK Supply and Use Tables 2022*. Office for National Statistics.
- BGS CMIC (2024). *Critical Minerals Intelligence Centre Reports*. British Geological Survey.
- BEIS (2023). *UK Critical Minerals Strategy*. Department for Energy Security and Net Zero.
- BGS (2025). *World Mineral Statistics 2025*. British Geological Survey.
- HMRC (2025). *UK Trade Statistics: Rare Earth Compounds (HS 2846)*. HM Revenue & Customs.
- Leontief, W. (1970). The dynamic inverse. In *Contributions to Input-Output Analysis*, North-Holland.
- Armington, P. S. (1969). A theory of demand for products distinguished by place of production. *IMF Staff Papers*, 16(1), 159–178.
- Ghosh, A. (1958). Input-output approach in an allocation system. *Economica*, 25(97), 58–64.
- Tesfatsion, L. & Judd, K. L. (Eds.) (2006). *Handbook of Computational Economics, Vol. 2: Agent-Based Computational Economics*. Elsevier.
- Farmer, J. D. et al. (2015). A third wave in the economics of climate change. *Environmental and Resource Economics*, 62(2), 329–357. [ABM-CGE integration methodology]
- Iori, G. & Mantegna, R. N. (2018). Empirical analyses of networks in finance. *Handbook of Computational Economics*, Vol. 4.
- All sources from REE_Supply_Chain_Risk_Analysis.md (Section 10), especially OECD ICIO 2025, ECB Economic Bulletin, and USGS OFR 2025-1047.

---
*Plan compiled: April 13, 2026 | Extends REE_Supply_Chain_Risk_Analysis.md for UK-focused DIO+CGE+ABM modelling*
