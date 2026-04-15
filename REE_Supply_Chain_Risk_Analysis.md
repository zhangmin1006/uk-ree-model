# Supply Chain and Risk Analysis Using Rare Earth Elements (REEs)
**Research Report | April 2026**

---

## 1. Research Objective and Analytical Framework

### 1.1 Objective

This study assesses how disruptions in the supply of rare earth elements (REEs)—arising from export controls, geopolitical tensions, and production shocks—propagate through industrial supply chains and affect downstream sectors and aggregate economic performance. The 2025 Chinese export restriction episode provides a real-time natural experiment validating the analytical framework.

### 1.2 Analytical Framework

The analysis integrates:

1. **Rare earth production and trade data** (USGS MCS 2025–2026; UN Comtrade HS 280530 / 2846)
2. **Input–output tables** (OECD ICIO 2025 edition; WIOD; EXIOBASE)
3. **Sector–REE mapping** (magnet-intensive sectors: EVs, wind turbines, defense, electronics)
4. **Supply-shock scenario design** (30% / 75% Chinese export reduction)
5. **IO-based supply constraint modeling** (Leontief + Ghosh hybrid)
6. **Quantification of output losses, risk exposure, and propagation paths**

---

## 2. Data Sources

### 2.1 Input–Output Tables

| Database | Coverage | REE Relevance |
|---|---|---|
| **OECD ICIO 2025** | 76 countries, 45 sectors, 1995–2020 | Best for cross-border REE risk transmission |
| **WIOD** | 43 countries, 56 sectors | Harmonized, widely used in academic REE studies |
| **EXIOBASE** | 44 countries + RoW, material extensions | Can integrate mineral extraction data; preferred for resource analysis |

For rare earth supply chain analysis, **EXIOBASE** is most appropriate because it includes material-flow extensions linking physical mineral inputs to economic sectors—enabling direct computation of REE intensity coefficients without sector-disaggregation proxies.

### 2.2 REE Production Data (USGS MCS 2025–2026)

**Global Mine Production (2025 estimate):** ~390,000 tonnes REO equivalent

| Country | Production (t REO) | Share (%) |
|---|---|---|
| China | ~270,000 | ~69 |
| United States | 51,000 | ~13 |
| Australia | ~18,000 | ~5 |
| Myanmar | ~38,000 | ~10 |
| Others | ~13,000 | ~3 |

**Global Reserves (2025):** ~110 million tonnes REO  
**U.S. Reserves:** 3.6 million tonnes (measured + indicated); >14 million tonnes in Canada

**U.S. Production Trend:**
- 2024: 45,000 t REO, valued at $260 million (Mountain Pass, CA)
- 2025: 51,000 t REO, valued at $240 million (price decline offset volume gains)

**U.S. Import Value:**
- 2024: $170 million (–11% YoY)
- 2025: $165 million (volume +169% as stockpiling accelerated; price decline offset surge)

### 2.3 REE Trade Data

**Chinese Export Volumes:**
- 2024: ~55,000 t REE shipped (+6% volume, but sharp value decline due to lower prices)
- 2024: ~58,000 t rare earth magnets exported
- May 2025: Chinese magnet exports fell **~75% year-over-year** following April 4 export controls

**U.S. Import Sources (2020–2025 average):**
- China: ~67–71% of consumption
- Malaysia, Japan, Estonia: remaining sources (primarily processing/re-export hubs for Chinese material)

**EU Import Sources (2024):**
- China: 46.3% by weight (6,000 t)
- Russia: 28.4% (3,700 t)
- Malaysia: 19.9% (2,600 t; primarily re-processed Chinese material)
- EU imports fell ~30% in 2024 due to demand decline and partial diversification

**Key HS Codes:**
- HS 280530: Rare earth metals (cerium, lanthanum, etc.)
- HS 2846 / 284690: Rare earth compounds (oxides, carbonates)

---

## 3. Mathematical Models

### 3.1 Standard Input–Output Model (Demand-Side Baseline)

The Leontief quantity model:

$$\mathbf{x} = (\mathbf{I} - \mathbf{A})^{-1} \mathbf{y}$$

Where:
- **x** = vector of total sectoral output (402 U.S. industries in USGS model; 45 sectors in OECD ICIO)
- **A** = matrix of direct input coefficients
- **y** = vector of final demand
- $(\mathbf{I} - \mathbf{A})^{-1}$ = Leontief inverse

This model quantifies how green-transition demand growth (EVs, wind) increases REE requirements upstream.

### 3.2 Supply-Side Shock Model

Because REE risks are primarily **supply-constrained**, a hybrid supply-restriction framework is required:

**Step 1 — Exogenous supply shock to sector k (REE production/exports):**

$$\Delta x_k = -\theta \cdot x_k, \quad 0 < \theta \leq 1$$

Scenario calibration:
- θ = 0.30: Moderate disruption (export control with partial licensing compliance)
- θ = 0.75: Severe disruption (observed May 2025 magnet export collapse)

**Step 2 — Ghosh (supply-driven) propagation:**

$$\mathbf{x}' = (\mathbf{I} - \mathbf{B})^{-1} \mathbf{v}$$

Where **B** is the output allocation matrix and **v** represents primary input supply.

**Step 3 — USGS Nonlinear Optimization Extension:**  
The USGS model minimizes deviations from pre-disruption interindustry demand across 402 industries, decomposing impacts into:
1. Consuming industries reducing output
2. Consuming industries paying higher prices
3. Domestic producing industries increasing output (partial offset)
4. Domestic producers receiving higher prices
5. Downstream industries reducing output (second-order effects)

### 3.3 REE Dependence Metrics

**Direct REE Dependence by sector i:**

$$RD_i = \frac{\text{REE inputs to sector } i}{\text{Total intermediate inputs of sector } i}$$

**Total (Direct + Indirect) REE Dependence:**

$$CRD = \mathbf{r}(\mathbf{I} - \mathbf{A})^{-1}$$

Where **r** is the row vector of REE inputs per unit of output. This captures *hidden* upstream dependencies not visible in direct sourcing.

**Application:** ECB analysis found >80% of large European firms are within **3 supply chain intermediaries** of a Chinese REE producer—demonstrating large indirect CRD values even for firms with no direct Chinese sourcing.

### 3.4 Supply Concentration: Herfindahl–Hirschman Index

$$HHI = \sum_j s_{ij}^2$$

Where $s_{ij}$ = share of REE imports to sector *i* from country *j*.

| Processing Stage | HHI | Interpretation |
|---|---|---|
| Global REE mining | ~5,000–6,000 | Monopoly territory (China ~69%) |
| Light REE separation | ~7,000–8,000 | Near-monopoly |
| Heavy REE separation (Dy, Tb) | ~9,500–10,000 | Effective monopoly |
| NdFeB magnet manufacturing | ~7,500–8,500 | China 85–90% global output |

*Note: U.S. competition law threshold for "highly concentrated" = HHI > 2,500. All REE processing stages far exceed this threshold. Recent research (ScienceDirect, 2025) notes HHI variation has larger price impacts at lower HHI values, questioning whether standard thresholds adequately capture the extreme tail-risk of near-monopoly markets.*

---

## 4. Sector–REE Mapping

### 4.1 Key REE Applications by Sector

| Sector | Primary REEs | Application | REE per Unit |
|---|---|---|---|
| Electric vehicles | Nd, Pr, Dy, Tb | NdFeB traction motors | 1–2 kg NdFeB magnets/EV |
| Offshore wind turbines | Nd, Pr, Dy, Tb | Direct-drive generators | Up to 4 t magnets/turbine |
| Defense (fighter jets) | Dy, Tb, Sm, Y | Guidance, avionics, heat coatings | F-35: >900 lbs REE/unit |
| Naval vessels | Multiple | Electronic systems, propulsion | Destroyer: ~5,200 lbs; Submarine: ~9,200 lbs |
| Consumer electronics | Nd, Eu, Y, Ce | Motors, displays, phosphors | ~0.1–0.5 kg/device |
| Petroleum refining | La, Ce | Fluid catalytic cracking | Continuous process input |
| LED/Phosphors | Eu, Tb, Y | Color rendering | Per kg phosphor |

### 4.2 IO Sector Mapping Procedure

Since REEs do not appear as a discrete sector in standard IO tables, the following disaggregation is required:
1. Allocate REE inputs to: **Non-ferrous metal mining**, **Basic metals**, **Specialized materials**
2. Use industry reports (IRENA, JRC, McKinsey) to derive sector-level REE intensity coefficients
3. Validate against engineering documentation and patent literature for NdFeB composition (Nd: 28.5%, Dy: 4.4%, B: 1%, Fe: 66%)

---

## 5. Disruption Scenarios and Simulated Impacts

### 5.1 Scenario Design

**Scenario A — Moderate Export Control (θ = 0.30)**
*Basis:* China requires export licenses for 7 heavy REEs (April 2025); partial compliance allows ~70% of normal volumes.

**Scenario B — Severe Supply Shock (θ = 0.75)**
*Basis:* Calibrated to observed May 2025 collapse in Chinese magnet exports (–75% YoY); equivalent to near-complete denial of access to Chinese HREEs.

**Scenario C — Sustained Restriction with Partial Substitution (θ = 0.50, 2-year duration)**
*Basis:* Policy scenario for planning purposes; accounts for 12-month suspension of October 2025 controls and gradual ex-China capacity ramp-up.

### 5.2 Estimated Economic Impacts

Based on USGS IO model results (402 U.S. industries, probability-weighted) and ECB supply chain network analysis:

**USGS Probability-Weighted GDP Impact (China disruption, 100% probability assigned):**

| REE Commodity | Estimated U.S. GDP Impact |
|---|---|
| Samarium | –$4,498 million |
| Lutetium | –$2,059 million |
| Terbium | –$1,809 million |
| Dysprosium | –$1,624 million |
| Other HREEs (Gd, Y, Sc, Eu) | –$500–800 million each (est.) |

*Note: ~50% of samarium impact derives from consuming industries absorbing higher prices (not output reduction), reflecting price inelasticity of REE-dependent production.*

**Observed 2025 Price Spikes (Post-April Controls):**

| Element | Price Increase |
|---|---|
| Yttrium | +598% |
| Terbium | +195% |
| Dysprosium | +168% |
| Samarium | +~6,000% (60× reported) |
| General EU rare earth prices | Up to 6× Chinese domestic price |

**Downstream Sector Observations (Scenario B-equivalent, May 2025):**
- European automotive plants halted or reduced utilization due to permanent magnet shortages
- EU sourced 98% of magnets from China; 100% of HREEs from China
- ~223 large European firms relied on a single supply chain intermediary for REE access
- 157 U.S. firms (including Microsoft, Apple, Intel) served as primary intermediary layer for EU manufacturers

### 5.3 Demand-Side Context: Green Transition Amplification

| Application | 2024 Demand (kt Nd-equiv.) | 2035 Projected Demand (kt) | CAGR |
|---|---|---|---|
| EV traction motors | 37 | ~110–130 | ~10–12% |
| Wind turbine generators | ~15 | ~40–50 | ~10% |
| Total magnetic REE demand | ~59 (2022 baseline) | 176 | ~10% |
| Market value (REO for energy) | $3.8B (2022) | $36.2B (2035) | 19.1% |

**Implication for IO modeling:** Rising final demand **y** for EVs and wind turbines propagates upstream through the Leontief inverse, increasing total REE requirements. With China's supply share effectively fixed, any demand growth without supply diversification raises HHI-risk exposure even without an active shock.

---

## 6. Critical Node Analysis

### 6.1 Industries with Highest REE Dependence

Based on direct (RD) and total (CRD) dependence metrics:

| Industry | Dependence Type | Key Vulnerability |
|---|---|---|
| NdFeB magnet manufacturing | Very high (direct) | 85–90% Chinese production; no near-term substitute |
| EV motor manufacturing | High (direct + indirect) | 1–2 kg NdFeB per vehicle; no substitute for high-torque applications |
| Offshore wind turbines | High (direct) | 4 t magnets per turbine; alternatives (electrically excited) require redesign |
| Defense electronics (F-35, ships) | High (direct + indirect) | Regulatory/supply chain security barriers limit substitution |
| Advanced electronics (Apple, Intel) | Moderate–high (indirect) | Multiple REE points of exposure; limited stockpiling capacity |
| Petroleum refining (FCC catalysts) | Moderate (direct) | La/Ce dependent; some substitution possible |

### 6.2 Amplification and Bottleneck Assessment

The most critical single bottleneck in the global REE supply chain is **heavy rare earth (HREE) separation and refining**, where:
- China controls **99% of global processing capacity** (Dy, Tb, Y processing)
- The first non-Chinese HREE separation facility (Lynas Malaysia) produced commercial-scale dysprosium oxide only in **May 2025**
- Non-Chinese HREE separation capacity is projected to remain **<20% of global total by 2028** under aggressive expansion scenarios

**Substitution limitations:**
- NdFeB magnets: No cost-competitive alternative for high-power-density applications (EVs, offshore wind)
- Dysprosium/Terbium addition: Required for high-temperature performance; reducing usage degrades efficiency
- SmCo magnets (samarium-cobalt): Alternative but higher cost, lower performance per kg

---

## 7. Policy Responses and Supply Diversification

### 7.1 Non-Chinese Supply Development (as of April 2026)

| Producer | Country | Capacity | Status |
|---|---|---|---|
| MP Materials (Mountain Pass) | USA | 51,000 t REO/yr + 1,000 t NdFeB magnets/yr | Operational; $400M U.S. government equity commitment |
| Lynas Rare Earths | Australia/Malaysia | ~7,000 t NdPr/yr (→12,000 t target) | Operational; HREE separation milestone May 2025 |
| Lynas USA (LREE + HREE facilities) | USA (TX) | Processing only (Hondo + Seadrift) | Under construction |
| Iluka Resources (Eneabba refinery) | Australia | 17,500 t/yr REO | Under construction |
| Arafura Nolans | Australia | 4,440 t NdPr/yr | Development stage |
| Noveon Magnetics | USA (TX) | Magnet manufacturing | Operational (small scale) |

**Gap assessment:** Even full build-out of all announced projects would cover only ~25–35% of current global demand outside China by 2028–2030.

### 7.2 Geopolitical Developments

- **April 4, 2025:** China imposes export licensing on 7 HREEs + magnets (Sm, Gd, Tb, Dy, Lu, Sc, Y)
- **October 9, 2025:** Controls expanded to 5 additional elements (Eu, Ho, Er, Tm, Yb) + technologies + extraterritorial reach (0.1% Chinese-origin threshold)
- **November 7, 2025:** October wave temporarily suspended until November 10, 2026 (April wave remains in force)
- **U.S. response:** $400M MP Materials equity; Apple $500M magnet agreement; U.S.-Australia $1B financing framework; expansion of National Defense Stockpile HREE holdings
- **EU response:** Critical Raw Materials Act; diversification targets; ECB supply chain mapping exercise

---

## 8. Interpretation of Results

The combined evidence from IO modeling, USGS scenario analysis, ECB network analysis, and market observation of the 2025 supply shock yields the following conclusions:

**Finding 1 — Extreme concentration creates systemic vulnerability.** HHI values of 7,000–10,000 for HREE processing place the global supply system far beyond any concentration threshold previously considered in competition policy. The 2025 price spikes (yttrium +598%, terbium +195%, samarium +6,000%) confirm that even partial export restriction triggers severe market dislocations.

**Finding 2 — Amplification effects are disproportionate to REE's economic share.** REEs account for a tiny share of GDP but provide enabling inputs to sectors (EVs, wind, defense, electronics) representing trillions in downstream value. A $165 million import market can trigger multi-billion-dollar output disruptions—the hallmark of a critical intermediate input.

**Finding 3 — Indirect dependencies dominate total exposure.** The ECB finding that >80% of large European firms are within 3 intermediaries of a Chinese REE producer illustrates why direct sourcing metrics (RD) systematically understate true vulnerability. Total dependence measures (CRD) using the Leontief inverse are essential.

**Finding 4 — Heavy REEs are the critical bottleneck.** Light REEs (Nd, Pr, La, Ce) have nascent non-Chinese supply chains. Heavy REEs (Dy, Tb, Y) have near-total Chinese processing monopoly. The green transition's reliance on HREE-enhanced magnets for high-temperature performance means demand is growing precisely where substitution is most difficult.

**Finding 5 — Supply diversification timelines are mismatched with risk horizon.** The 2025 shock demonstrated real near-term risk. Even optimistic projections place non-Chinese HREE capacity at <20% of global total by 2028. The gap between risk materialization and supply diversification creates a multi-year structural vulnerability window.

---

## 9. Strengths and Limitations

### Strengths
- Captures both direct and indirect supply chain transmission
- Validated against real 2025 shock episode
- Integrates physical (production/trade) and economic (IO) data
- Applicable to national (U.S., EU) and sector-specific policy analysis

### Limitations
- Standard IO models assume **fixed technical coefficients** (no substitution)
- Cannot model **inventory drawdown** or **stockpiling buffers**
- Limited ability to differentiate 17 individual REEs in standard IO tables without manual disaggregation
- **Data gap:** Element-specific tonnage for Chinese production is not publicly disaggregated
- **Dynamic adjustment** (price-induced switching, redesign) not captured

### Recommended Extensions
1. **CGE integration** for medium-term supply response and substitution modeling
2. **Network analysis** (combined with IO) for propagation pathway mapping
3. **Stochastic scenarios** with machine-learning-derived probability weights (as per USGS 2025 methodology)
4. **Element-level disaggregation** for LREE vs. HREE differentiation
5. **Joint analysis** with carbon transition policies (EV mandate timelines drive HREE demand trajectory)

---

## 10. Key Data Tables for Further Analysis

### Table 1: Global REE Production by Country (2025 Estimates)
| Country | Production (t REO) | Share | Primary Elements |
|---|---|---|---|
| China | 270,000 | 69% | All, esp. ionic clay HREEs |
| USA | 51,000 | 13% | LREE (Nd, Pr) at Mountain Pass |
| Myanmar | ~38,000 | ~10% | HREE ionic clay |
| Australia | ~18,000 | ~5% | LREE (Nd, Pr, La, Ce) |
| India | ~2,900 | ~1% | Coastal placer deposits |
| Russia | ~2,700 | ~1% | Multiple |
| Others | ~8,000 | ~2% | Varied |
| **Total** | **~390,000** | **100%** | |

### Table 2: REE Processing Concentration
| Processing Stage | China Share | HHI (approx.) |
|---|---|---|
| Mining | 69% | ~5,500 |
| Light REE separation | ~85% | ~7,500 |
| Heavy REE separation | ~99% | ~9,800 |
| NdFeB magnet manufacturing | ~85–90% | ~7,500–8,500 |
| Dysprosium oxide production | ~98% | ~9,600 |

### Table 3: Downstream Sector Exposure Summary
| Sector | HREE Exposure | Substitutability | Disruption Risk |
|---|---|---|---|
| EV manufacturing | High (Dy, Tb) | Very low (near-term) | Critical |
| Offshore wind | High (Dy, Tb) | Low | Critical |
| Defense electronics | High (Dy, Tb, Y) | Very low (strategic) | Critical |
| Consumer electronics | Moderate | Low–moderate | High |
| Industrial motors | Moderate | Low | High |
| Petroleum refining | Low–moderate (La, Ce) | Moderate | Moderate |
| Medical devices | Moderate (Y, Gd) | Low | High |

---

## Sources

- [USGS Mineral Commodity Summaries 2026 — Rare Earths](https://pubs.usgs.gov/periodicals/mcs2026/mcs2026-rare-earths.pdf)
- [USGS Mineral Commodity Summaries 2025 — Rare Earths](https://pubs.usgs.gov/periodicals/mcs2025/mcs2025-rare-earths.pdf)
- [USGS Open File Report 2025-1047: IO Model Methodology for Critical Minerals](https://pubs.usgs.gov/publication/ofr20251047/full)
- [IEA — With new export controls on critical minerals, supply concentration risks become reality](https://www.iea.org/commentaries/with-new-export-controls-on-critical-minerals-supply-concentration-risks-become-reality)
- [ECB Economic Bulletin Focus — Euro area vulnerability to Chinese rare earth export restrictions](https://www.ecb.europa.eu/press/economic-bulletin/focus/2025/html/ecb.ebbox202506_01~44d432008e.en.html)
- [CSIS — The Consequences of China's New Rare Earths Export Restrictions](https://www.csis.org/analysis/consequences-chinas-new-rare-earths-export-restrictions)
- [CSIS — Developing Rare Earth Processing Hubs: An Analytical Approach](https://www.csis.org/analysis/developing-rare-earth-processing-hubs-analytical-approach)
- [Global Policy Watch — Heavy Rare Earth Elements: Rising Supply Chain Risks and Emerging Policy Responses](https://www.globalpolicywatch.com/2026/02/heavy-rare-earth-elements-rising-supply-chain-risks-and-emerging-policy-responses/)
- [European Parliament Think Tank — China's Rare Earth Export Restrictions](https://epthinktank.eu/2025/11/24/chinas-rare-earth-export-restrictions/)
- [Chatham House — China's new restrictions on rare earth exports send a stark warning to the West](https://www.chathamhouse.org/2025/10/chinas-new-restrictions-rare-earth-exports-send-stark-warning-west)
- [ScienceDirect — Evaluating criticality of strategic metals: HHI and usual concentration thresholds](https://www.sciencedirect.com/science/article/pii/S0140988325000313)
- [ScienceDirect — Geopolitical risk and global supply of rare earth permanent magnets](https://www.sciencedirect.com/science/article/pii/S0140988325003202)
- [OECD — Inter-Country Input-Output (ICIO) Tables and TiVA Database, 2025 Edition](https://www.oecd.org/en/blogs/2025/11/oecd-inter-country-input-output-icio-tables-and-trade-in-value-added-tiva-database-launch-of-the-2025-edition.html)
- [Eurostat — Imports of rare earth elements saw 30% drop in 2024](https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20250409-1)
- [S&P Global — Rare earth supply bottlenecks set to persist in 2026](https://www.spglobal.com/energy/en/news-research/latest-news/metals/012726-rare-earth-supply-bottlenecks-set-to-persist-in-2026)
- [OEC — Rare Earth Metal Compounds (HS 2846) Trade Profile](https://oec.world/en/profile/hs/rare-earth-metal-compounds)
- [IRENA — Critical Materials for the Energy Transition: Rare Earth Elements](https://www.irena.org/-/media/Irena/Files/Technical-papers/IRENA_Rare_Earth_Elements_2022.pdf)
- [JRC — The Role of Rare Earth Elements in Wind Energy and Electric Mobility](https://publications.jrc.ec.europa.eu/repository/bitstream/JRC122671/jrc122671_the_role_of_rare_earth_elements_in_wind_energy_and_electric_mobility_2.pdf)
- [McKinsey — Powering the energy transition's motor: Circular rare earth elements](https://www.mckinsey.com/industries/metals-and-mining/our-insights/powering-the-energy-transitions-motor-circular-rare-earth-elements)

---
*Report generated: April 13, 2026 | Methodology follows CLAUDE.md framework for REE supply chain and IO-based risk analysis*
