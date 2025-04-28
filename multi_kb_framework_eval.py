#!/usr/bin/env python3
"""
Multi-KB Framework Evaluation - Tests the dynamic knowledge system framework
across a diverse set of complex problems from different domains.

This script runs the framework on 10 challenging problems across different industries
and domains, and evaluates performance, architecture selection, and solution quality.
"""

import os
import json
import asyncio
import logging
import time
import argparse
from typing import Dict, List, Any, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from dynamic_multi_kb_framework import DynamicFrameworkAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("framework_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("framework-eval")

# Define evaluation test cases across diverse domains
EVAL_PROBLEMS = [
    # 1. Legal domain problem
    {
        "id": "legal_compliance",
        "name": "Legal Compliance Framework",
        "domain": "Legal",
        "problem": """
        Design a comprehensive legal compliance framework for a multinational technology company 
        that handles user data across jurisdictions including the EU (GDPR), California (CCPA), 
        China (PIPL), and emerging global standards. The framework should address data collection, 
        processing, storage, transfer, breach notification, and user rights while minimizing 
        operational friction and allowing for regional variations in a unified corporate strategy.
        """
    },
    
    # 2. Financial domain problem
    {
        "id": "esg_investment",
        "name": "ESG Investment Strategy",
        "domain": "Finance",
        "problem": """
        Develop an Environmental, Social, and Governance (ESG) investment strategy for a 
        pension fund that balances financial returns with sustainability goals. Consider 
        climate transition risks, social impact metrics, governance standards, regulatory 
        trends, data quality issues, greenwashing concerns, and the integration of these 
        factors into traditional financial analysis across different asset classes and 
        geographic regions.
        """
    },
    
    # 3. Machine learning research problem
    {
        "id": "ml_explainability",
        "name": "ML Model Explainability",
        "domain": "AI Research",
        "problem": """
        Design a comprehensive framework for explaining complex machine learning models 
        (including deep neural networks) that balances technical accuracy with interpretability 
        for different stakeholders. The solution should address local and global explanations, 
        feature importance, counterfactual reasoning, causal relationships, technical and 
        non-technical user needs, visualization techniques, and regulatory requirements for 
        model transparency across high-stakes domains like healthcare and finance.
        """
    },
    
    # 4. Web technology problem
    {
        "id": "web_architecture",
        "name": "Web Architecture Design",
        "domain": "Web Technology",
        "problem": """
        Design a scalable web architecture for a global e-commerce platform that needs to handle 
        25 million daily active users, peak loads during shopping events, real-time inventory, 
        personalized recommendations, secure payments, and multi-region deployment. The solution 
        should address frontend performance, backend scalability, database architecture, caching 
        strategies, CDN implementation, microservices organization, API design, security practices, 
        and observability while considering cost efficiency and development team productivity.
        """
    },
    
    # 5. Healthcare problem
    {
        "id": "healthcare_interoperability",
        "name": "Healthcare Interoperability",
        "domain": "Healthcare",
        "problem": """
        Create a comprehensive strategy for achieving healthcare data interoperability across 
        a fragmented ecosystem of hospitals, clinics, insurance providers, and digital health 
        platforms. The solution should address technical standards (FHIR, HL7), data governance, 
        privacy concerns, patient consent, legacy system integration, real-time data access, 
        identification and matching challenges, and incentive alignment while ensuring clinical 
        workflows aren't disrupted and patient care is improved through better information sharing.
        """
    },
    
    # 6. Environmental problem
    {
        "id": "climate_adaptation",
        "name": "Climate Adaptation Strategy",
        "domain": "Environmental",
        "problem": """
        Develop a climate change adaptation strategy for coastal cities facing rising sea levels, 
        increased storm intensity, and changing precipitation patterns. The solution should address 
        infrastructure resilience, natural buffer zones, emergency response systems, vulnerable 
        population protection, economic impacts, funding mechanisms, governance approaches, and 
        phased implementation while integrating scientific projections, community values, and 
        uncertainty management in decision-making processes.
        """
    },
    
    # 7. Education problem
    {
        "id": "education_transformation",
        "name": "Education Digital Transformation",
        "domain": "Education",
        "problem": """
        Design a digital transformation strategy for higher education institutions adapting to 
        changing student needs, technological capabilities, and workforce requirements. The solution 
        should address online/hybrid learning models, personalized education pathways, credential 
        innovation, lifelong learning support, faculty development, technology infrastructure, 
        data-driven decision making, and sustainable business models while maintaining educational 
        quality, accessibility, and institutional values in a rapidly evolving landscape.
        """
    },
    
    # 8. Cybersecurity problem
    {
        "id": "cyber_security",
        "name": "Enterprise Cybersecurity Strategy",
        "domain": "Cybersecurity",
        "problem": """
        Develop an enterprise cybersecurity strategy for a multinational organization with legacy 
        systems, cloud services, IoT deployments, and a hybrid workforce. The solution should address 
        threat modeling, defense-in-depth architecture, identity management, data protection, detection 
        and response capabilities, supply chain risks, compliance requirements, security awareness, 
        and incident response planning while balancing security controls with business operations 
        and optimizing resource allocation based on risk prioritization.
        """
    },
    
    # 9. Supply chain problem
    {
        "id": "supply_chain",
        "name": "Resilient Supply Chain Design",
        "domain": "Supply Chain",
        "problem": """
        Design a resilient and sustainable global supply chain strategy that can withstand disruptions 
        from pandemics, geopolitical conflicts, climate events, and other systemic shocks. The solution 
        should address network diversification, inventory optimization, supplier relationship management, 
        visibility and transparency, scenario planning, rapid response mechanisms, environmental impact 
        reduction, and ethical labor practices while maintaining cost efficiency and service levels in 
        both normal and disrupted conditions.
        """
    },
    
    # 10. Urban planning problem
    {
        "id": "smart_city",
        "name": "Smart City Implementation",
        "domain": "Urban Planning",
        "problem": """
        Develop a comprehensive smart city implementation plan for a mid-sized city seeking to improve 
        quality of life, operational efficiency, and sustainability. The solution should address 
        technology infrastructure (IoT, data platforms, connectivity), priority application areas 
        (mobility, energy, public safety, services), data governance, privacy protection, digital 
        inclusion, public-private partnerships, citizen engagement, and performance measurement while 
        ensuring solutions are interoperable, scalable, and provide tangible benefits to all residents.
        """
    }
]


class FrameworkEvaluator:
    """Evaluates the Dynamic Multi-KB Framework across diverse problem domains"""
    
    def __init__(self, output_dir: str = "./eval_results"):
        """Initialize the evaluator"""
        self.framework = None
        self.results = []
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize the framework"""
        self.framework = DynamicFrameworkAPI()
        await self.framework.initialize()
        logger.info("Framework initialized for evaluation")
    
    async def evaluate_problem(self, problem_data: Dict[str, Any], 
                             run_all_modes: bool = False) -> Dict[str, Any]:
        """
        Evaluate the framework on a specific problem.
        
        Args:
            problem_data: Problem definition and metadata
            run_all_modes: Whether to run all operation modes
            
        Returns:
            Evaluation results
        """
        problem_id = problem_data["id"]
        problem_statement = problem_data["problem"].strip()
        
        logger.info(f"Evaluating problem: {problem_id}")
        
        # Analyze the problem
        start_time = time.time()
        analysis = await self.framework.analyze_problem(problem_statement)
        analysis_time = time.time() - start_time
        
        # Record analysis results
        result = {
            "problem_id": problem_id,
            "problem_name": problem_data["name"],
            "domain": problem_data["domain"],
            "analysis": {
                "architecture": analysis["architecture"],
                "operation_mode": analysis["operation_mode"],
                "complexity": analysis["complexity"]["overall"],
                "relevant_domains": analysis["relevant_domains"][:5],
                "characteristics": [k for k, v in analysis["characteristics"].items() if v],
                "analysis_time": analysis_time
            },
            "solutions": []
        }
        
        # Determine which modes to run
        if run_all_modes:
            modes = ["collaborative", "competitive", "emergent"]
        else:
            modes = [analysis["operation_mode"]]
        
        # Solve with each mode
        for mode in modes:
            try:
                logger.info(f"Solving {problem_id} with {mode} mode")
                
                # Solve the problem
                start_time = time.time()
                solution = await self.framework.solve_problem(problem_statement, mode=mode)
                solve_time = time.time() - start_time
                
                # Record solution
                solution_result = {
                    "mode": mode,
                    "solution": solution["solution"],
                    "domains_utilized": solution["domains_utilized"],
                    "solve_time": solve_time,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Error solving {problem_id} with {mode} mode: {e}")
                
                # Record error
                solution_result = {
                    "mode": mode,
                    "error": str(e),
                    "success": False
                }
            
            result["solutions"].append(solution_result)
        
        return result
    
    async def run_evaluation(self, problems: List[Dict[str, Any]] = None, 
                           run_all_modes: bool = False):
        """
        Run evaluation on multiple problems.
        
        Args:
            problems: List of problems to evaluate (defaults to EVAL_PROBLEMS)
            run_all_modes: Whether to run all operation modes
        """
        if not self.framework:
            await self.initialize()
        
        if problems is None:
            problems = EVAL_PROBLEMS
        
        for problem in problems:
            result = await self.evaluate_problem(problem, run_all_modes)
            self.results.append(result)
            
            # Save individual result
            result_path = Path(self.output_dir) / f"{problem['id']}_result.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Completed evaluation for problem: {problem['id']}")
        
        # Save all results
        all_results_path = Path(self.output_dir) / "all_results.json"
        with open(all_results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate reports
        self.generate_reports()
        
        logger.info(f"Evaluation complete. Results saved to {self.output_dir}")
    
    def generate_reports(self):
        """Generate evaluation reports and visualizations"""
        if not self.results:
            logger.warning("No results to generate reports from")
            return
        
        # Generate summary statistics
        self._generate_summary_stats()
        
        # Generate architecture selection report
        self._generate_architecture_report()
        
        # Generate performance report
        self._generate_performance_report()
        
        # Generate domain utilization report
        self._generate_domain_report()
    
    def _generate_summary_stats(self):
        """Generate summary statistics"""
        # Create summary data
        summary = {
            "total_problems": len(self.results),
            "architectures_selected": {},
            "modes_selected": {},
            "avg_analysis_time": 0,
            "avg_solve_time": 0,
            "success_rate": 0
        }
        
        total_analysis_time = 0
        total_solve_time = 0
        successful_solutions = 0
        total_solutions = 0
        
        # Collect statistics
        for result in self.results:
            # Count architectures
            arch = result["analysis"]["architecture"]
            if arch not in summary["architectures_selected"]:
                summary["architectures_selected"][arch] = 0
            summary["architectures_selected"][arch] += 1
            
            # Count operation modes
            mode = result["analysis"]["operation_mode"]
            if mode not in summary["modes_selected"]:
                summary["modes_selected"][mode] = 0
            summary["modes_selected"][mode] += 1
            
            # Sum times
            total_analysis_time += result["analysis"]["analysis_time"]
            
            # Count solutions
            for solution in result["solutions"]:
                total_solutions += 1
                if solution.get("success", False):
                    successful_solutions += 1
                    total_solve_time += solution.get("solve_time", 0)
        
        # Calculate averages
        if len(self.results) > 0:
            summary["avg_analysis_time"] = total_analysis_time / len(self.results)
        
        if successful_solutions > 0:
            summary["avg_solve_time"] = total_solve_time / successful_solutions
        
        if total_solutions > 0:
            summary["success_rate"] = successful_solutions / total_solutions
        
        # Save summary
        summary_path = Path(self.output_dir) / "summary_stats.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate textual report
        report = [
            "# Dynamic Multi-KB Framework Evaluation Summary",
            "",
            f"Total problems evaluated: {summary['total_problems']}",
            f"Overall solution success rate: {summary['success_rate']*100:.1f}%",
            f"Average problem analysis time: {summary['avg_analysis_time']:.2f} seconds",
            f"Average problem solving time: {summary['avg_solve_time']:.2f} seconds",
            "",
            "## Architecture Selection",
            ""
        ]
        
        for arch, count in summary["architectures_selected"].items():
            report.append(f"- {arch}: {count} problems ({count/summary['total_problems']*100:.1f}%)")
        
        report.extend([
            "",
            "## Operation Mode Selection",
            ""
        ])
        
        for mode, count in summary["modes_selected"].items():
            report.append(f"- {mode}: {count} problems ({count/summary['total_problems']*100:.1f}%)")
        
        # Save report
        report_path = Path(self.output_dir) / "summary_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report))
    
    def _generate_architecture_report(self):
        """Generate architecture selection report"""
        # Create data for report
        data = []
        
        for result in self.results:
            row = {
                "Problem": result["problem_name"],
                "Domain": result["domain"],
                "Architecture": result["analysis"]["architecture"],
                "Mode": result["analysis"]["operation_mode"],
                "Complexity": f"{result['analysis']['complexity']:.2f}",
                "Characteristics": ", ".join(result["analysis"]["characteristics"][:3])
            }
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = Path(self.output_dir) / "architecture_selection.csv"
        df.to_csv(csv_path, index=False)
        
        # Create table for markdown
        table = tabulate(df, headers="keys", tablefmt="pipe")
        
        report = [
            "# Architecture Selection Analysis",
            "",
            "This report shows the architecture and operation mode selected for each problem,",
            "along with the problem complexity and key characteristics that influenced the decision.",
            "",
            table
        ]
        
        # Save report
        report_path = Path(self.output_dir) / "architecture_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report))
        
        # Generate visualization
        try:
            # Architecture distribution chart
            plt.figure(figsize=(10, 6))
            arch_counts = df["Architecture"].value_counts()
            arch_counts.plot(kind="bar", color="skyblue")
            plt.title("Architecture Selection Distribution")
            plt.xlabel("Architecture")
            plt.ylabel("Number of Problems")
            plt.tight_layout()
            plt.savefig(Path(self.output_dir) / "architecture_distribution.png")
            
            # Mode selection chart
            plt.figure(figsize=(10, 6))
            mode_counts = df["Mode"].value_counts()
            mode_counts.plot(kind="bar", color="lightgreen")
            plt.title("Operation Mode Distribution")
            plt.xlabel("Mode")
            plt.ylabel("Number of Problems")
            plt.tight_layout()
            plt.savefig(Path(self.output_dir) / "mode_distribution.png")
        except Exception as e:
            logger.error(f"Error creating architecture visualizations: {e}")
    
    def _generate_performance_report(self):
        """Generate performance report"""
        # Create data for report
        data = []
        
        for result in self.results:
            for solution in result["solutions"]:
                if solution.get("success", False):
                    row = {
                        "Problem": result["problem_name"],
                        "Domain": result["domain"],
                        "Mode": solution["mode"],
                        "Analysis_Time_s": result['analysis'].get('analysis_time', 0),
                        "Solve_Time_s": solution.get('solve_time', 0),
                        "Domains_Used": len(solution.get("domains_utilized", []))
                    }
                    data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = Path(self.output_dir) / "performance_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        # Create nicer column names for display
        display_df = df.copy()
        display_df.columns = ["Problem", "Domain", "Mode", "Analysis Time (s)", "Solve Time (s)", "Domains Used"]
        
        # Create table for markdown
        table = tabulate(display_df, headers="keys", tablefmt="pipe")
        
        report = [
            "# Performance Metrics Report",
            "",
            "This report shows the performance metrics for each successful problem solution,",
            "including analysis time, solution time, and the number of knowledge domains utilized.",
            "",
            table
        ]
        
        # Save report
        report_path = Path(self.output_dir) / "performance_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report))
        
        # Generate visualization
        try:
            # Performance by domain chart
            plt.figure(figsize=(12, 8))
            domain_perf = df.groupby("Domain")[["Analysis_Time_s", "Solve_Time_s"]].mean()
            domain_perf.plot(kind="bar", figsize=(12, 6))
            plt.title("Average Performance by Domain")
            plt.xlabel("Domain")
            plt.ylabel("Time (seconds)")
            plt.tight_layout()
            plt.savefig(Path(self.output_dir) / "performance_by_domain.png")
            
            # Performance by mode chart
            plt.figure(figsize=(10, 6))
            mode_perf = df.groupby("Mode")[["Analysis_Time_s", "Solve_Time_s"]].mean()
            mode_perf.plot(kind="bar", figsize=(10, 6))
            plt.title("Average Performance by Operation Mode")
            plt.xlabel("Operation Mode")
            plt.ylabel("Time (seconds)")
            plt.tight_layout()
            plt.savefig(Path(self.output_dir) / "performance_by_mode.png")
        except Exception as e:
            logger.error(f"Error creating performance visualizations: {e}")
    
    def _generate_domain_report(self):
        """Generate domain utilization report"""
        # Collect domain usage data
        domain_usage = {}
        
        for result in self.results:
            for solution in result["solutions"]:
                if solution.get("success", False):
                    for domain in solution.get("domains_utilized", []):
                        if domain not in domain_usage:
                            domain_usage[domain] = {
                                "count": 0,
                                "problems": []
                            }
                        
                        domain_usage[domain]["count"] += 1
                        
                        if result["problem_id"] not in domain_usage[domain]["problems"]:
                            domain_usage[domain]["problems"].append(result["problem_id"])
        
        # Sort domains by usage count
        sorted_domains = sorted(domain_usage.items(), key=lambda x: x[1]["count"], reverse=True)
        
        # Create report
        report = [
            "# Knowledge Domain Utilization Report",
            "",
            "This report shows the utilization of knowledge domains across all problems,",
            "sorted by frequency of use.",
            "",
            "| Domain | Usage Count | Problems |",
            "|--------|-------------|----------|"
        ]
        
        for domain, data in sorted_domains:
            problems_str = ", ".join(data["problems"])
            report.append(f"| {domain} | {data['count']} | {problems_str} |")
        
        # Save report
        report_path = Path(self.output_dir) / "domain_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report))
        
        # Generate visualization
        try:
            # Top domains chart
            plt.figure(figsize=(12, 8))
            top_domains = dict(sorted_domains[:20])  # Top 20 domains
            domain_names = [d for d in top_domains.keys()]
            domain_counts = [d["count"] for d in top_domains.values()]
            
            plt.barh(domain_names, domain_counts, color="coral")
            plt.title("Top Knowledge Domains by Utilization")
            plt.xlabel("Number of Uses")
            plt.tight_layout()
            plt.savefig(Path(self.output_dir) / "top_domains.png")
        except Exception as e:
            logger.error(f"Error creating domain visualization: {e}")
    
    async def close(self):
        """Close the framework"""
        if self.framework:
            await self.framework.shutdown()
            logger.info("Framework shut down")


async def run_evaluation(problems_to_run: List[str] = None, run_all_modes: bool = False):
    """
    Run the framework evaluation.
    
    Args:
        problems_to_run: IDs of specific problems to run (runs all if None)
        run_all_modes: Whether to run all operation modes
    """
    # Filter problems if specific ones are requested
    if problems_to_run:
        problems = [p for p in EVAL_PROBLEMS if p["id"] in problems_to_run]
        if not problems:
            logger.error(f"No matching problems found for IDs: {problems_to_run}")
            return
    else:
        problems = EVAL_PROBLEMS
    
    # Create evaluator
    evaluator = FrameworkEvaluator()
    
    try:
        # Run evaluation
        await evaluator.run_evaluation(problems, run_all_modes)
        
        logger.info(f"Evaluation complete. Tested {len(problems)} problems.")
        
        # Print summary location
        print(f"\nEvaluation complete!")
        print(f"Results and reports saved to: {os.path.abspath(evaluator.output_dir)}")
        print("Key reports:")
        print(f"  - Summary: {os.path.abspath(os.path.join(evaluator.output_dir, 'summary_report.md'))}")
        print(f"  - Architecture selection: {os.path.abspath(os.path.join(evaluator.output_dir, 'architecture_report.md'))}")
        print(f"  - Performance metrics: {os.path.abspath(os.path.join(evaluator.output_dir, 'performance_report.md'))}")
        print(f"  - Domain utilization: {os.path.abspath(os.path.join(evaluator.output_dir, 'domain_report.md'))}")
        
    finally:
        # Close evaluator
        await evaluator.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-KB Framework Evaluation")
    parser.add_argument("--problems", type=str, nargs="+", 
                       help="Specific problem IDs to evaluate")
    parser.add_argument("--all-modes", action="store_true", 
                       help="Run all operation modes for each problem")
    parser.add_argument("--list-problems", action="store_true",
                       help="List available problems without running evaluation")
    
    args = parser.parse_args()
    
    if args.list_problems:
        print("Available evaluation problems:")
        print("{:<20} {:<30} {:<15}".format("ID", "Name", "Domain"))
        print("-" * 65)
        for problem in EVAL_PROBLEMS:
            print("{:<20} {:<30} {:<15}".format(
                problem["id"], problem["name"], problem["domain"]
            ))
    else:
        # Run evaluation
        asyncio.run(run_evaluation(args.problems, args.all_modes))