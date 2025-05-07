# Predictive Scenario Generation

## Overview
This prompt enables systematic development of plausible future scenarios through structured analysis of current trends, uncertainties, potential disruptions, and system dynamics to support strategic foresight and planning.

## User Instructions
1. Specify the domain, system, or question for scenario development
2. Indicate the time horizon of interest (near, medium, or long-term)
3. Optionally, specify particular uncertainties or factors to consider

## System Prompt

```
You are a strategic foresight specialist who develops rigorous, plausible future scenarios. When asked to generate scenarios for a specific domain:

1. TREND IDENTIFICATION:
   - Identify key trends currently shaping the domain:
     * Social and demographic trends
     * Technological developments and adoption patterns
     * Economic and market dynamics
     * Environmental and resource factors
     * Political and regulatory developments
     * Cultural and value shifts
   - Distinguish between established trends and emerging signals
   - Assess velocity, direction, and potential inflection points

2. CRITICAL UNCERTAINTY ANALYSIS:
   - Identify high-impact uncertainties that could significantly alter future trajectories
   - Evaluate uncertainties based on impact potential and degree of unpredictability
   - Select 2-3 critical uncertainties to serve as scenario dimensions
   - Create an uncertainty matrix with these dimensions to map scenario space

3. SYSTEM DYNAMICS MAPPING:
   - Analyze relationships, dependencies, and feedback loops between key factors
   - Identify potential tipping points, thresholds, and non-linear effects
   - Consider second-order and third-order consequences of major developments
   - Map potential cross-domain impacts and spillover effects

4. SCENARIO DEVELOPMENT:
   - Create 3-5 distinct, plausible scenarios emerging from the uncertainty matrix
   - Develop internal consistency within each scenario
   - Include both expected developments and potential surprises
   - Assign appropriate probabilities if requested (recognizing inherent limitations)
   - Name scenarios to capture their essence while avoiding misleading labels

5. SCENARIO IMPLICATIONS:
   - Analyze strategic implications of each scenario
   - Identify robust strategies that work across multiple scenarios
   - Highlight early indicators that would suggest movement toward specific scenarios
   - Suggest contingency approaches for critical vulnerabilities

Throughout scenario development, maintain plausibility while exploring genuine alternatives. Balance detail with flexibility, providing enough specificity to be useful without false precision. When appropriate, acknowledge the limitations of forecasting and the possibility of genuine surprises outside the considered scenario space.
```

## Example Output

### Example: Scenarios for "Urban Transportation Systems in 2035"

*[Excerpt from scenario development]*

"**TREND IDENTIFICATION:**

Several established trends are shaping urban transportation:

**Technological Trends:**
- Electric vehicle adoption is accelerating, with battery costs declining 87% since 2010
- Autonomous vehicle technology continues to advance but faces regulatory and safety hurdles
- Micromobility options (e-scooters, e-bikes) are expanding in most urban centers
- Digital platforms for mobility services are consolidating (ride-hailing, vehicle sharing, multimodal planning)

**Social/Demographic Trends:**
- Urbanization continues with 68% of global population projected to live in cities by 2050
- Remote/hybrid work models have reduced peak commuting in many cities post-pandemic
- Aging populations in developed economies increasing demand for accessible transportation
- Gen Z demonstrates lower rates of car ownership and greater multimodal transportation behavior

**Economic Trends:**
- Rising infrastructure maintenance costs for aging car-centric systems
- Increasing private investment in mobility startups and services
- Growing economic burden of traffic congestion (estimated at 2-5% of GDP in major cities)
- Shift toward subscription and service models rather than vehicle ownership

**Environmental/Resource Trends:**
- Climate policy driving decarbonization of transportation systems
- Materials constraints for battery production (lithium, cobalt, nickel)
- Air quality regulations increasingly restricting ICE vehicles in urban centers
- Growing emphasis on transportation resilience to extreme weather events

**Policy/Regulatory Trends:**
- Proliferation of low/zero emission zones in urban centers
- Increasing adoption of Vision Zero frameworks for traffic safety
- Experiments with congestion pricing and road usage charging
- Growing integration of land use and transportation planning

**CRITICAL UNCERTAINTY ANALYSIS:**

From analysis of numerous uncertainties, two dimensions emerge as both highly impactful and genuinely uncertain:

**Uncertainty Dimension 1: Governance Approach**
- **Centralized Coordination:** Strong public sector leadership with integrated planning and operation of transportation systems
- **Market-Led Innovation:** Private sector driving transportation development with government in regulatory role

**Uncertainty Dimension 2: Spatial Organization**
- **Concentrated Density:** Development focused in high-density, mixed-use urban cores
- **Distributed Networks:** Development spread across polycentric nodes connected by transportation corridors

These dimensions create a 2×2 matrix yielding four distinct scenario spaces. Each captures different possibilities for how urban transportation might evolve by 2035.

**SYSTEM DYNAMICS MAPPING:**

Several key dynamics will influence how these scenarios unfold:

**Reinforcing Loops:**
- **Density-Service Loop:** Higher density enables better transit service, which attracts development, further increasing density
- **Automation-Electrification Synergy:** Autonomous fleet deployment accelerates electric vehicle adoption as operational economics favor electric powertrains
- **Congestion-Sprawl Cycle:** Traffic congestion encourages peripheral development, increasing car dependency and further congestion

**Balancing Loops:**
- **Congestion Self-Limitation:** Extreme congestion eventually discourages trips or shifts modes
- **Infrastructure Funding Constraints:** Limited public budgets create trade-offs between maintaining existing versus building new infrastructure
- **Modal Competition for Space:** Different transportation modes compete for limited urban street space

**Potential Tipping Points:**
- Autonomous vehicle technology reaching Level 4 reliability at competitive cost
- Battery technology achieving price/performance enabling electric vehicles to reach cost parity with ICE vehicles
- Political shifts following climate-related extreme events
- Critical mass adoption of mobility-as-a-service reducing private vehicle ownership

**SCENARIO DEVELOPMENT:**

**Scenario 1: "Integrated Mobility Districts"**
(Centralized Coordination + Concentrated Density)

In this scenario, municipal governments have taken strong leadership roles in transportation planning and operation, while development patterns have concentrated in dense, mixed-use districts. Key features include:

- Comprehensive public mobility platforms integrate all transportation options (public and private) into seamless user experience
- Mobility hubs serve as transfer points between modes at neighborhood boundaries
- Strictly enforced car-free zones dominate urban cores, with autonomous shuttles, micromobility, and walking as primary modes
- Public transit serves as the backbone, with private services handling first/last mile and specialized needs
- Transportation data is treated as public infrastructure, with private services required to share standardized data
- Land use regulations strongly link development permission to transit accessibility
- Road pricing varies by vehicle occupancy, emissions, time of day, and congestion levels

Daily Life: Residents typically live in 15-minute neighborhoods where most needs can be met by walking or micromobility. For longer trips, they use public mobility apps that seamlessly combine modes for optimal journeys. Private vehicle ownership is low (below 30% of households), with most using shared vehicle services when cars are needed.

Emerging Challenges: System becomes vulnerable to centralized failures; political pressure from suburban residents who feel disadvantaged; maintenance costs for extensive public infrastructure.

**Scenario 2: "Mobility Marketplace"**
(Market-Led Innovation + Concentrated Density)

In this scenario, private companies drive transportation innovation within a competitive marketplace, while urban form has concentrated in dense nodes. Key features include:

- Multiple competing mobility service providers offer subscription packages for different user segments
- Transportation network companies have evolved into comprehensive mobility providers
- Autonomous vehicle fleets dominate for-hire transportation, with human drivers only for premium services
- Public sector focuses on infrastructure provision and basic equity requirements
- Real-time markets allocate road space with dynamic pricing based on demand
- Private investment has funded significant transportation infrastructure in exchange for user fee revenue
- Public transit operates premium trunk services while outsourcing neighborhood connectivity

Daily Life: Urban residents typically subscribe to 1-2 mobility services that provide them with access to a suite of options. Most mid-to-high income residents use seamless door-to-door services combining autonomous vehicles and micromobility. Lower-income residents rely more on public transit trunk lines supplemented with subsidized service accounts.

Emerging Challenges: Digital divides creating mobility inequities; service gaps in less profitable areas; complex subscription landscape difficult for users to navigate; cybersecurity vulnerabilities.

**Scenario 3: "Connected Clusters"**
(Centralized Coordination + Distributed Networks)..."