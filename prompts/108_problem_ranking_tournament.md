# Problem-Ranking Tournament

## Overview
This prompt template establishes a rigorous methodology for systematically comparing and prioritizing customer problems based on impact and solvability. Using a tournament-style elimination framework, it forces explicit trade-off decisions between competing problems to identify the most valuable opportunities for product development. This structured approach overcomes common biases in problem selection and ensures resources are directed toward opportunities with the highest potential return.

## User Instructions
Provide information about:
1. The target customer segment or persona
2. A list of known problems or pain points (at least 5-10)
3. Any relevant business constraints or capabilities
4. Your current hypotheses about which problems might be most important
5. Any existing data about problem frequency or severity

## System Prompt

I'll conduct a comprehensive Problem-Ranking Tournament to systematically evaluate and prioritize customer problems for product development. This structured methodology follows a rigorous framework:

### 1. Problem Statement Refinement
- Rewrite each problem as a clear, specific, and measurable statement
- Standardize format: "[User] struggles to [action] because [obstacle], resulting in [consequence]"
- Eliminate duplicate problems and consolidate related issues
- Split compound problems into distinct atomic challenges
- Clarify ambiguous terminology and validate shared understanding

### 2. Evaluation Criteria Development
For each problem, I will assess:

**Impact Dimensions:**
- **Pain Frequency**: How often users encounter this problem (daily, weekly, monthly)
- **Pain Intensity**: Severity of the negative experience when encountered (1-10 scale)
- **Functional Impact**: Degree to which the problem prevents job completion
- **Emotional Impact**: Level of frustration, anxiety, or other negative emotions generated
- **Financial Impact**: Direct or indirect costs imposed by the problem
- **Market Size**: Percentage of target users who experience this problem
- **Trend Trajectory**: Whether this problem is becoming more or less significant over time

**Solvability Dimensions:**
- **Technical Feasibility**: Level of technical challenge to implement a solution
- **Resource Requirements**: Expected time, cost, and team size needed
- **Solution Clarity**: How well-defined the potential solution approach is
- **Competitive Differentiation**: Uniqueness of solution compared to alternatives
- **Integration Complexity**: Effort required to fit solution into existing ecosystem
- **Regulatory Barriers**: Legal or compliance challenges to implementing solution
- **Time to Value**: How quickly a solution could deliver meaningful benefits

### 3. Evidence Collection & Assessment
For each problem, I will:
- Identify existing quantitative evidence supporting impact claims
- Note qualitative insights from customer interactions and feedback
- Highlight gaps in current understanding requiring further research
- Assess confidence level in our understanding of the problem
- Identify potential measurement approaches for validation

### 4. Tournament Structure Setup
- Organize problems into balanced brackets for head-to-head comparison
- Create seeding based on initial hypothesis of problem importance
- Design multi-round elimination structure leading to final rankings
- Establish rules for advancement and tiebreaking scenarios
- Configure tournament structure based on problem count and evaluation needs

### 5. Preliminary Scoring Round
For each problem:
- Score independently on each impact and solvability dimension (1-10)
- Calculate weighted composite scores for total impact and solvability
- Position each problem on an Impact-Solvability matrix
- Flag problems with high variance or uncertainty for deeper analysis
- Create initial rank order based on combined scoring

### 6. Head-to-Head Comparative Analysis
For each matchup:
- Directly compare two problems across all evaluation dimensions
- Force explicit trade-off decisions when dimensions conflict
- Document specific reasoning for each advancement decision
- Identify critical differentiating factors in close comparisons
- Update confidence assessments based on comparative analysis

### 7. Championship Round Analysis
For finalist problems:
- Conduct comprehensive SWOT analysis for each problem
- Identify potential solution approaches and implementation paths
- Map interdependencies with other high-ranking problems
- Project expected value creation if successfully solved
- Document minority opinions for problems that narrowly missed finals

### 8. Final Prioritization Framework
- Produce rank-ordered list of problems based on tournament results
- Cluster problems into priority tiers (Critical, High, Medium, Low)
- Map problem relationships and potential solution synergies
- Identify quick wins (high impact, high solvability, low resource needs)
- Flag strategic problems requiring longer-term investment

### 9. Validation & Research Plan
- Design validation approach for top-ranked problems
- Identify specific metrics to measure actual impact
- Develop research questions to address knowledge gaps
- Create testing methodology to validate problem significance
- Establish timeline for ongoing problem reassessment

### 10. Implementation Roadmap
- Create problem-solution mapping for top priorities
- Develop phased approach to addressing problem clusters
- Establish resource requirements for addressing top problems
- Define success metrics for problem resolution
- Create timeline for solution development and deployment

The completed Problem-Ranking Tournament will provide a rigorously prioritized list of customer problems with clear justification for selected priorities, specific metrics for validation, and a strategic roadmap for solution development.

## Example Usage

### User Input
Target customer: Small business owners (5-20 employees) in retail
Known problems:
1. Struggle with inventory management across multiple locations
2. Difficulty hiring and retaining qualified staff
3. Cash flow challenges due to seasonal sales fluctuations
4. Limited visibility into which products are most profitable
5. Time-consuming reconciliation of in-store and online sales
6. Challenges competing with larger retailers on price
7. Difficulty building and maintaining an online presence
8. Managing vendor relationships and negotiating terms
9. Limited marketing budget and expertise
10. Inconsistent customer experience across channels

Business constraints: Limited technical team (2 developers), $250K development budget
Current hypothesis: Inventory management and profitability tracking are most critical
Existing data: Survey showing 78% report inventory issues, 63% struggle with profitability analysis

### System Response

# Problem-Ranking Tournament Results: Small Retail Business Owners

## 1. Problem Statement Refinement

**Refined Problem Statements:**

1. **Inventory Visibility**: Small retail business owners struggle to maintain accurate inventory counts across physical and online channels because existing systems don't integrate seamlessly, resulting in overselling, stockouts, and reduced customer satisfaction.

2. **Staff Management**: Small retailers struggle to attract, train, and retain qualified staff because they can't compete with larger retailers' compensation packages, resulting in inconsistent store operations and excessive owner time spent on hiring.

3. **Cash Flow Management**: Small retailers struggle to manage cash flow during seasonal fluctuations because they lack sophisticated financial forecasting tools, resulting in missed opportunities and financial stress during slow periods.

4. **Profitability Analysis**: Small retailers struggle to identify their most profitable products and categories because their point-of-sale and accounting systems don't provide granular margin analysis, resulting in suboptimal purchasing and pricing decisions.

5. **Sales Channel Reconciliation**: Small retailers waste 5-10 hours weekly manually reconciling transactions between in-store POS and e-commerce platforms because these systems don't automatically integrate, resulting in accounting errors and reduced time for strategic activities.

6. **Price Competitiveness**: Small retailers struggle to price products competitively because they lack real-time competitor pricing data and automated repricing capabilities, resulting in lost sales to larger retailers or unnecessary margin erosion.

7. **Online Presence Management**: Small retailers struggle to maintain compelling online storefronts because they lack technical expertise and time to manage multiple digital channels, resulting in inconsistent branding and missed sales opportunities.

8. **Vendor Relationship Management**: Small retailers struggle to optimize vendor terms and discover new suppliers because they lack systematic approaches to vendor evaluation and negotiation, resulting in suboptimal purchasing terms and limited product selection.

9. **Marketing Efficiency**: Small retailers struggle to generate return on their limited marketing budget because they lack data-driven targeting capabilities, resulting in inefficient ad spend and inconsistent customer acquisition.

10. **Omnichannel Experience**: Small retailers fail to deliver consistent customer experiences across in-store, online, and mobile touchpoints because their systems and processes are fragmented, resulting in customer confusion and reduced loyalty.

## 2. Tournament Structure & Preliminary Scoring

**Impact Scoring (1-10 scale)**

| Problem | Frequency | Intensity | Functional | Emotional | Financial | Market Size | Trajectory | Impact Score |
|---------|-----------|-----------|------------|-----------|-----------|-------------|------------|--------------|
| Inventory Visibility | 9 | 8 | 9 | 7 | 8 | 9 | 8 | 8.3 |
| Staff Management | 7 | 8 | 7 | 9 | 7 | 8 | 9 | 7.9 |
| Cash Flow Management | 6 | 9 | 8 | 10 | 10 | 7 | 7 | 8.1 |
| Profitability Analysis | 8 | 7 | 9 | 6 | 9 | 8 | 8 | 7.9 |
| Sales Channel Reconciliation | 10 | 6 | 7 | 8 | 5 | 7 | 9 | 7.4 |
| Price Competitiveness | 8 | 7 | 6 | 8 | 8 | 6 | 8 | 7.3 |
| Online Presence Management | 7 | 6 | 8 | 7 | 8 | 9 | 10 | 7.9 |
| Vendor Relationship Management | 5 | 6 | 7 | 5 | 8 | 7 | 6 | 6.3 |
| Marketing Efficiency | 7 | 7 | 7 | 7 | 8 | 9 | 9 | 7.7 |
| Omnichannel Experience | 8 | 7 | 8 | 7 | 7 | 8 | 10 | 7.9 |

**Solvability Scoring (1-10 scale, 10 = most solvable)**

| Problem | Technical | Resources | Solution Clarity | Differentiation | Integration | Regulation | Time to Value | Solvability Score |
|---------|-----------|-----------|------------------|-----------------|-------------|------------|---------------|------------------|
| Inventory Visibility | 7 | 6 | 8 | 8 | 7 | 9 | 7 | 7.4 |
| Staff Management | 5 | 4 | 5 | 6 | 8 | 7 | 5 | 5.7 |
| Cash Flow Management | 6 | 5 | 7 | 7 | 7 | 8 | 6 | 6.6 |
| Profitability Analysis | 8 | 7 | 9 | 9 | 8 | 10 | 8 | 8.4 |
| Sales Channel Reconciliation | 8 | 7 | 9 | 8 | 6 | 9 | 8 | 7.9 |
| Price Competitiveness | 7 | 8 | 7 | 7 | 7 | 8 | 7 | 7.3 |
| Online Presence Management | 6 | 5 | 7 | 5 | 6 | 9 | 6 | 6.3 |
| Vendor Relationship Management | 7 | 8 | 6 | 8 | 8 | 8 | 7 | 7.4 |
| Marketing Efficiency | 6 | 5 | 6 | 7 | 7 | 9 | 6 | 6.6 |
| Omnichannel Experience | 5 | 4 | 6 | 8 | 5 | 8 | 5 | 5.9 |

**Combined Opportunity Score (Impact × Solvability)**

1. Profitability Analysis: 66.4
2. Inventory Visibility: 61.4
3. Sales Channel Reconciliation: 58.5
4. Price Competitiveness: 53.3
5. Cash Flow Management: 53.5
6. Marketing Efficiency: 50.8
7. Vendor Relationship Management: 46.6
8. Online Presence Management: 49.8
9. Omnichannel Experience: 46.6
10. Staff Management: 45.0

## 3. Head-to-Head Tournament Results

**Round 1 Matchups:**
1. Inventory Visibility vs. Vendor Management → Inventory Visibility advances
2. Profitability Analysis vs. Staff Management → Profitability Analysis advances
3. Cash Flow Management vs. Marketing Efficiency → Cash Flow Management advances
4. Sales Channel Reconciliation vs. Online Presence → Sales Channel Reconciliation advances
5. Price Competitiveness vs. Omnichannel Experience → Price Competitiveness advances

**Round 2 Matchups:**
1. Profitability Analysis vs. Price Competitiveness → Profitability Analysis advances
2. Inventory Visibility vs. Sales Channel Reconciliation → Inventory Visibility advances
3. Cash Flow Management vs. (Bye) → Cash Flow Management advances

**Semi-Final Matchups:**
1. Profitability Analysis vs. Cash Flow Management → Profitability Analysis advances
2. Inventory Visibility vs. (Bye) → Inventory Visibility advances

**Championship Analysis:**
Profitability Analysis vs. Inventory Visibility

**Critical Comparison Factors:**
1. Both problems scored highly on impact (7.9 vs. 8.3)
2. Profitability Analysis has significantly higher solvability (8.4 vs. 7.4)
3. Profitability Analysis requires less integration complexity
4. Inventory management requires more real-time data processing
5. Profitability insights enable better inventory decisions
6. Solving profitability analysis creates foundation for addressing other problems
7. Implementation complexity favors Profitability Analysis given resource constraints

**WINNER: Profitability Analysis**
Profitability Analysis edges out Inventory Visibility due to higher solvability, faster time-to-value, and its enabling effect on other problem solutions, including inventory management.

## 4. Final Problem Prioritization

**Tier 1 (Critical Priority):**
1. **Profitability Analysis** - Enables data-driven decisions across business
2. **Inventory Visibility** - Highest overall impact score with good solvability

**Tier 2 (High Priority):**
3. **Sales Channel Reconciliation** - High frequency, excellent solvability
4. **Cash Flow Management** - Extreme emotional and financial impact
5. **Price Competitiveness** - Balanced scores with significant market pressure

**Tier 3 (Medium Priority):**
6. **Marketing Efficiency** - Important but requires more complex solutions
7. **Online Presence Management** - Growing importance but resource-intensive
8. **Vendor Relationship Management** - Good solvability but lower relative impact

**Tier 4 (Lower Priority):**
9. **Omnichannel Experience** - Important strategically but complex implementation
10. **Staff Management** - High impact but low solvability within software solution

## 5. Solution Approach for Top Priority

**Profitability Analysis Solution Concept:**

**Core Functionality:**
- Product-level margin calculation integrating COGS, operational costs, and pricing
- Category and vendor profitability visualization with trend analysis
- Automated data integration from POS and inventory systems
- Intelligent product mix recommendations based on margin analysis
- What-if scenario modeling for pricing and promotion decisions
- Benchmarking against industry standards by product category
- Automatic identification of profit leakage points

**Implementation Phases:**

**Phase 1: Foundation (1-2 months)**
- Basic data integration with existing POS and accounting systems
- Product-level margin calculation dashboard
- Simple profitability reporting by category and time period
- Data validation and reconciliation tools

**Phase 2: Enhanced Analytics (3-4 months)**
- Vendor profitability analysis and comparison tools
- Promotion impact assessment on product margins
- Intelligent product mix optimization suggestions
- Inventory turns correlated with profitability metrics

**Phase 3: Advanced Optimization (5-6 months)**
- Dynamic pricing recommendations based on profitability goals
- Predictive analytics for seasonal profit optimization
- Customer segment profitability analysis
- Advanced what-if scenario modeling for business decisions

**Success Metrics:**
- 20% reduction in time spent on financial analysis
- 15% increase in average product margin through better decision-making
- 10% reduction in slow-moving, low-profit inventory
- 25% improvement in promotion ROI through data-driven planning

**Resource Requirements:**
- 1.5 developers for 6 months (phased implementation)
- $120,000 of $250,000 budget (48%)
- Integration with existing POS and accounting systems
- Minimal ongoing maintenance with automated data procedures

## 6. Validation Plan

**Key Validation Activities:**

1. **Prototype Testing (Weeks 1-4)**
   - Create dashboard mockups with simulated data
   - Conduct usability testing with 5-7 small retailers
   - Validate that insights would change actual business decisions
   - Refine specific metrics and analysis views based on feedback

2. **Data Integration Validation (Weeks 3-6)**
   - Test API connections with common POS and accounting systems
   - Validate data accuracy between source systems and analysis platform
   - Measure system performance with realistic data volumes
   - Identify potential data quality issues requiring resolution

3. **Economic Impact Assessment (Ongoing)**
   - Select 3-5 retailers for beta implementation
   - Establish baseline profitability metrics before implementation
   - Track specific business decisions influenced by the system
   - Measure financial impact of those decisions after 60 days

**Success Criteria:**
- 90%+ of test users can identify their least profitable products within 5 minutes
- System provides actionable recommendations that users wouldn't have identified otherwise
- Data reconciliation requires less than 30 minutes per week of user intervention
- Beta users report at least 3 profit-improving decisions made using the system

## 7. Secondary Problem Integration

The Profitability Analysis solution can be designed to address aspects of other high-priority problems:

**Inventory Visibility Enhancement:**
- Integrate profitability metrics with inventory levels to identify high-profit, at-risk stockout items
- Flag slow-moving inventory with low profitability for potential discounting
- Calculate optimal reorder points based on profitability and turn rates

**Cash Flow Management Support:**
- Project cash requirements based on optimal inventory levels of high-profit items
- Identify seasonal trends in product profitability to inform cash planning
- Provide data for negotiating better vendor payment terms based on profit contribution

**Sales Channel Optimization:**
- Compare profitability across online vs. in-store channels
- Identify products that should be channel-exclusive based on margin analysis
- Calculate true cost-to-serve for different fulfillment methods

This Problem-Ranking Tournament provides a rigorous foundation for product development, clearly identifying Profitability Analysis as the highest-value opportunity with a concrete implementation plan optimized for available resources and maximum business impact.