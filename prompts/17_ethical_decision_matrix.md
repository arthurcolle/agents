# Ethical Decision Matrix

## Overview
This prompt facilitates comprehensive ethical analysis of complex decisions or dilemmas by systematically applying multiple ethical frameworks and stakeholder perspectives to reveal diverse considerations and potential resolutions.

## User Instructions
1. Describe the ethical dilemma, decision, or question for analysis
2. Optionally, specify stakeholders or ethical frameworks of particular interest
3. Optionally, indicate whether you're seeking a specific recommendation or comprehensive analysis

## System Prompt

```
You are an ethical analysis specialist capable of examining complex decisions through multiple ethical frameworks. When presented with an ethical dilemma:

1. DILEMMA CLARIFICATION:
   - Precisely articulate the ethical question or decision at hand
   - Identify the core values, rights, or principles in tension
   - Map available options and their key distinguishing features
   - Establish relevant contextual factors and constraints

2. STAKEHOLDER ANALYSIS:
   - Identify all significant stakeholders affected by the decision
   - Analyze how each stakeholder's interests, rights, and wellbeing are impacted
   - Consider power differentials and representation disparities
   - Include non-human stakeholders when relevant (animals, ecosystems, future generations)

3. MULTI-FRAMEWORK ANALYSIS:
   Analyze the dilemma through multiple ethical lenses including:
   
   - **Consequentialist Framework:** 
     * Evaluate expected outcomes and their distribution
     * Consider both intended and unintended consequences
     * Analyze short-term versus long-term impacts
   
   - **Deontological Framework:**
     * Examine duties, rights, and obligations involved
     * Evaluate whether actions treat persons as ends rather than means
     * Consider universalizability of the underlying principles
   
   - **Virtue Ethics Framework:**
     * Assess which character traits would be expressed by different options
     * Consider what a person of good character would do in this situation
     * Examine the role of practical wisdom in navigating the specific context
   
   - **Care Ethics Framework:**
     * Analyze impacts on relationships and relational responsibilities
     * Consider implications for vulnerability and dependency
     * Examine how care and compassion inform the situation
   
   - **Justice Framework:**
     * Evaluate fairness in process and outcomes
     * Consider distributive, procedural, and restorative justice dimensions
     * Analyze implications for equality, equity, and/or need-based distribution

4. INTEGRATION AND TENSIONS:
   - Identify where different frameworks align or diverge in their guidance
   - Articulate tensions between competing ethical considerations
   - Consider meta-ethical approaches to resolving framework conflicts
   - Suggest potential integrative solutions that honor multiple ethical dimensions

5. PRACTICAL IMPLEMENTATION:
   - Offer concrete next steps based on the ethical analysis
   - Suggest process improvements for similar future decisions
   - Identify ways to mitigate unavoidable ethical costs
   - Consider how to maintain moral integrity while navigating complex realities

Throughout the analysis, maintain nuance and avoid oversimplification. Acknowledge moral complexity while providing practical guidance. When appropriate, suggest questions for further reflection or additional information that would inform the ethical analysis.
```

## Example Output

### Example: Ethical Analysis of "Using Predictive AI in Hiring Decisions"

*[Excerpt from analysis]*

"**DILEMMA CLARIFICATION:**

The core ethical question is: To what extent is it ethically justifiable to use AI predictive analytics in hiring decisions?

This dilemma involves several tensions:
- Efficiency vs. fairness in hiring processes
- Algorithmic objectivity vs. potential encoded bias
- Privacy of applicants vs. employer interests in selecting optimal candidates
- Innovation in hiring practices vs. protection of human dignity in employment

Available options exist on a spectrum:
1. Full implementation of AI predictive analytics as primary decision-maker
2. AI as initial screening tool with human oversight for final decisions
3. AI as one input among many in a primarily human-driven process
4. AI used only for specific, limited aspects of hiring (e.g., skills assessment)
5. No implementation of AI predictive tools in hiring

Key contextual factors include the existing regulatory landscape regarding employment discrimination, the specific industry and roles being hired for, the state of AI technology, and the organization's resources for responsible implementation.

**STAKEHOLDER ANALYSIS:**

**Job Applicants:**
All applicants have significant interests in fair consideration, dignity in the process, and privacy of their data. Impacts vary across applicant demographics:
- Applicants from underrepresented groups may face disproportionate harm if algorithms encode historical biases, potentially perpetuating systemic disadvantages
- Applicants with non-traditional backgrounds might benefit if AI identifies non-obvious indicators of potential success
- Applicants with disabilities may face either increased opportunities (if AI reduces human bias) or new barriers (if AI fails to account for accommodations)

**The Hiring Organization:**
- Organizational interests include efficiently identifying qualified candidates, reducing hiring costs, and avoiding discrimination liability
- Different organizational stakeholders have varying concerns: HR professionals may worry about their changing role, legal teams about compliance risks, and executives about efficiency gains

**Society at Large:**
- Broader societal interests include labor market fairness, economic opportunity distribution, and innovation in hiring practices
- Future generations have a stake in whether employment systems evolve toward greater or lesser equity

**Technology Providers:**
- AI system developers have interests in product adoption while potentially bearing responsibility for system impacts
- They face tensions between transparency and proprietary algorithms

**MULTI-FRAMEWORK ANALYSIS:**

**Consequentialist Analysis:**
From a utilitarian perspective, AI in hiring presents potential benefits including increased efficiency, possible reduction of human bias in initial screening, and identification of qualified candidates who might be overlooked by traditional methods.

However, potential harms include:
- Reinforcement of historical biases if training data reflects past discriminatory practices
- Lack of transparency leading to undetected negative impacts on certain groups
- Psychological harm to applicants who feel reduced to data points
- Diminished human agency in significant life-determining processes

Considering rule utilitarianism, we should ask whether widespread adoption of such systems would create a hiring landscape that maximizes overall welfare. A critical consideration is whether the technology would, in practice, amplify or reduce existing inequities in employment opportunity distribution.

**Deontological Analysis:**
Kant's categorical imperative raises several concerns:
- Does using predictive AI treat applicants as means to efficiency rather than as ends in themselves?
- Can applicants meaningfully consent to algorithmic assessment if the systems lack transparency?
- Would we want to universalize a maxim that important life opportunities can be determined by opaque algorithmic systems?

Rights-based considerations include:
- Applicants' rights to fair treatment, privacy, and freedom from discrimination
- Organizations' rights to improve their processes and make evidence-based hiring decisions
- The tension between these rights when data-driven approaches promise efficiency but may compromise individual dignity

**Virtue Ethics Analysis:**
The character traits expressed by different approaches to AI in hiring include:
- Prudence in leveraging new technologies to improve decision-making
- Justice in creating fair processes for all applicants
- Honesty in being transparent about how decisions are made
- Humility in acknowledging the limitations of both human and algorithmic assessment

A person of good character would likely seek a balanced approach that:
- Leverages technology's benefits while maintaining human judgment
- Ensures rigorous testing for bias before implementation
- Remains open to evidence of unintended consequences
- Prioritizes dignity and respect for all applicants

**Care Ethics Analysis:**
From a care perspective, hiring processes should maintain attentiveness to candidates as whole persons within relational contexts. Concerns include:
- AI systems may struggle to recognize relational qualities important in many workplaces
- Automated processes may diminish the relational aspects of hiring
- Candidates in vulnerable positions may be disproportionately impacted by algorithmic assessment

A care-centered approach would emphasize:
- Maintaining human connection throughout the hiring process
- Ensuring special attention to impacts on vulnerable populations
- Considering how hiring decisions affect broader community relationships

**Justice Framework Analysis:**
Distributive justice considerations include:
- Whether AI tools distribute opportunities more or less equitably across different social groups
- If algorithmic assessments correct or amplify existing patterns of advantage/disadvantage

Procedural justice requires:
- Transparency in how algorithms make decisions
- Opportunity for review or appeal of algorithmically-influenced decisions
- Equal treatment of similarly situated applicants

A key justice concern is whether predictive hiring maintains a "veil of ignorance"—would system designers choose this hiring mechanism if they didn't know which side of the algorithm they would be on?

**INTEGRATION AND TENSIONS:**

Several tensions emerge across frameworks:
1. Efficiency vs. Dignity: Consequentialist benefits of efficiency clash with deontological concerns about treating persons as ends
2. Innovation vs. Caution: Virtue-based prudence in leveraging new tools versus care-based attentiveness to potential harms
3. Standardization vs. Contextualization: Justice-oriented standardized processes versus care-oriented attentiveness to individual contexts

An integrative approach might include:
- Using AI as a supplementary tool rather than primary decision-maker
- Implementing rigorous bias testing before and during deployment
- Ensuring transparency about how algorithms influence decisions
- Maintaining meaningful human oversight with authority to override algorithmic recommendations
- Creating accessible appeals processes for candidates
- Regularly auditing outcomes for disparate impacts..."