#!/usr/bin/env python3
"""
AILES Legal AI - Data Quality Checker
Comprehensive validation of training data quality
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Comprehensive data quality validation"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.components = ['chatbot', 'predictor', 'explainer']
        self.quality_report = {}
    
    def check_all_components(self) -> Dict[str, Any]:
        """Check quality of all components"""
        logger.info("🔍 Starting comprehensive data quality check...")
        
        overall_stats = {
            'total_examples': 0,
            'total_files': 0,
            'components': {},
            'quality_issues': [],
            'recommendations': []
        }
        
        for component in self.components:
            file_path = self.data_dir / f"{component}_training_data.jsonl"
            
            if file_path.exists():
                logger.info(f"📊 Checking {component} data...")
                component_stats = self.check_component_quality(component, file_path)
                overall_stats['components'][component] = component_stats
                overall_stats['total_examples'] += component_stats['example_count']
                overall_stats['total_files'] += 1
            else:
                logger.warning(f"❌ Missing file: {file_path}")
                overall_stats['quality_issues'].append(f"Missing {component} training data")
        
        # Generate overall recommendations
        overall_stats['recommendations'] = self._generate_recommendations(overall_stats)
        
        return overall_stats
    
    def check_component_quality(self, component: str, file_path: Path) -> Dict[str, Any]:
        """Check quality of individual component"""
        stats = {
            'component': component,
            'file_path': str(file_path),
            'example_count': 0,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'valid_json_count': 0,
            'invalid_examples': [],
            'schema_violations': [],
            'content_quality': {},
            'diversity_metrics': {},
            'sample_examples': []
        }
        
        examples = []
        line_number = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        example = json.loads(line)
                        examples.append(example)
                        stats['valid_json_count'] += 1
                    except json.JSONDecodeError as e:
                        stats['invalid_examples'].append({
                            'line': line_number,
                            'error': str(e),
                            'content': line[:100] + "..." if len(line) > 100 else line
                        })
        
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return stats
        
        stats['example_count'] = len(examples)
        
        if examples:
            # Schema validation
            stats['schema_violations'] = self._validate_schema(component, examples)
            
            # Content quality analysis
            stats['content_quality'] = self._analyze_content_quality(component, examples)
            
            # Diversity metrics
            stats['diversity_metrics'] = self._calculate_diversity_metrics(component, examples)
            
            # Sample examples for manual review
            stats['sample_examples'] = examples[:3] if len(examples) >= 3 else examples
        
        return stats
    
    def _validate_schema(self, component: str, examples: List[Dict]) -> List[Dict]:
        """Validate schema compliance"""
        violations = []
        required_fields = ['instruction', 'input', 'output']
        
        for i, example in enumerate(examples):
            # Check required fields
            for field in required_fields:
                if field not in example:
                    violations.append({
                        'example_index': i,
                        'type': 'missing_field',
                        'field': field,
                        'severity': 'critical'
                    })
            
            # Validate output JSON structure for structured responses
            if 'output' in example:
                try:
                    output_data = json.loads(example['output']) if isinstance(example['output'], str) else example['output']
                    
                    # Component-specific validation
                    if component == 'chatbot':
                        if not isinstance(output_data, dict) or 'response' not in output_data:
                            violations.append({
                                'example_index': i,
                                'type': 'invalid_chatbot_structure',
                                'expected': 'JSON with "response" field',
                                'severity': 'high'
                            })
                    elif component == 'predictor':
                        if not isinstance(output_data, dict) or 'predicted_outcome' not in output_data:
                            violations.append({
                                'example_index': i,
                                'type': 'invalid_predictor_structure',
                                'expected': 'JSON with "predicted_outcome" field',
                                'severity': 'high'
                            })
                    elif component == 'explainer':
                        if not isinstance(output_data, dict) or 'detailed_legal_analysis' not in output_data:
                            violations.append({
                                'example_index': i,
                                'type': 'invalid_explainer_structure',
                                'expected': 'JSON with "detailed_legal_analysis" field',
                                'severity': 'high'
                            })
                
                except (json.JSONDecodeError, TypeError):
                    violations.append({
                        'example_index': i,
                        'type': 'invalid_json_output',
                        'severity': 'critical'
                    })
        
        return violations
    
    def _analyze_content_quality(self, component: str, examples: List[Dict]) -> Dict[str, Any]:
        """Analyze content quality metrics"""
        quality_metrics = {
            'avg_instruction_length': 0,
            'avg_input_length': 0,
            'avg_output_length': 0,
            'empty_content_count': 0,
            'very_short_content_count': 0,
            'very_long_content_count': 0,
            'language_quality': {},
            'legal_terminology_count': 0
        }
        
        instruction_lengths = []
        input_lengths = []
        output_lengths = []
        
        legal_terms = [
            'court', 'judgment', 'legal', 'law', 'act', 'section', 'case',
            'family law', 'divorce', 'custody', 'maintenance', 'property',
            'inheritance', 'adoption', 'domestic violence', 'child arrangements'
        ]
        
        for example in examples:
            # Length analysis
            instruction_len = len(example.get('instruction', ''))
            input_len = len(example.get('input', ''))
            output_len = len(str(example.get('output', '')))
            
            instruction_lengths.append(instruction_len)
            input_lengths.append(input_len)
            output_lengths.append(output_len)
            
            # Quality checks
            if instruction_len == 0 or input_len == 0 or output_len == 0:
                quality_metrics['empty_content_count'] += 1
            
            if instruction_len < 20 or input_len < 10 or output_len < 10:
                quality_metrics['very_short_content_count'] += 1
            
            if instruction_len > 1000 or input_len > 2000 or output_len > 5000:
                quality_metrics['very_long_content_count'] += 1
            
            # Legal terminology check
            full_text = f"{example.get('instruction', '')} {example.get('input', '')} {str(example.get('output', ''))}".lower()
            legal_term_count = sum(1 for term in legal_terms if term in full_text)
            if legal_term_count > 0:
                quality_metrics['legal_terminology_count'] += 1
        
        # Calculate averages
        if examples:
            quality_metrics['avg_instruction_length'] = sum(instruction_lengths) / len(instruction_lengths)
            quality_metrics['avg_input_length'] = sum(input_lengths) / len(input_lengths)
            quality_metrics['avg_output_length'] = sum(output_lengths) / len(output_lengths)
            quality_metrics['legal_content_percentage'] = (quality_metrics['legal_terminology_count'] / len(examples)) * 100
        
        return quality_metrics
    
    def _calculate_diversity_metrics(self, component: str, examples: List[Dict]) -> Dict[str, Any]:
        """Calculate diversity and variety metrics"""
        diversity_metrics = {
            'unique_instructions': 0,
            'unique_inputs': 0,
            'input_variety_score': 0.0,
            'case_type_distribution': {},
            'complexity_distribution': {},
            'qualification_distribution': {},
            'scenario_patterns': []
        }
        
        instructions = [example.get('instruction', '') for example in examples]
        inputs = [example.get('input', '') for example in examples]
        
        # Basic uniqueness
        diversity_metrics['unique_instructions'] = len(set(instructions))
        diversity_metrics['unique_inputs'] = len(set(inputs))
        diversity_metrics['input_variety_score'] = len(set(inputs)) / len(examples) if examples else 0
        
        # Component-specific analysis
        if component == 'chatbot':
            # Analyze qualification distribution
            qualifications = []
            case_types = []
            
            for example in examples:
                try:
                    output_data = json.loads(example.get('output', '{}'))
                    if 'qualification' in output_data:
                        qualifications.append(output_data['qualification'])
                    if 'case_type_detected' in output_data:
                        case_types.append(output_data['case_type_detected'])
                except:
                    continue
            
            diversity_metrics['qualification_distribution'] = dict(Counter(qualifications))
            diversity_metrics['case_type_distribution'] = dict(Counter(case_types))
        
        elif component == 'predictor':
            # Analyze complexity distribution
            complexities = []
            
            for example in examples:
                try:
                    output_data = json.loads(example.get('output', '{}'))
                    if 'complexity_assessment' in output_data and 'level' in output_data['complexity_assessment']:
                        complexities.append(output_data['complexity_assessment']['level'])
                except:
                    continue
            
            diversity_metrics['complexity_distribution'] = dict(Counter(complexities))
        
        # Scenario pattern analysis
        input_patterns = []
        for input_text in inputs[:50]:  # Sample first 50 for pattern analysis
            # Extract key patterns
            if 'divorce' in input_text.lower():
                input_patterns.append('divorce_scenario')
            elif 'child' in input_text.lower():
                input_patterns.append('child_scenario')
            elif 'property' in input_text.lower():
                input_patterns.append('property_scenario')
            elif 'inherit' in input_text.lower():
                input_patterns.append('inheritance_scenario')
            else:
                input_patterns.append('other_scenario')
        
        diversity_metrics['scenario_patterns'] = dict(Counter(input_patterns))
        
        return diversity_metrics
    
    def _generate_recommendations(self, overall_stats: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        for component, stats in overall_stats['components'].items():
            example_count = stats['example_count']
            
            # Quantity recommendations
            if example_count < 1000:
                recommendations.append(f"🔢 Consider generating more {component} examples (current: {example_count}, recommended: 2000+)")
            
            # Quality recommendations
            if stats['schema_violations']:
                critical_violations = sum(1 for v in stats['schema_violations'] if v['severity'] == 'critical')
                if critical_violations > 0:
                    recommendations.append(f"🚨 Fix {critical_violations} critical schema violations in {component} data")
            
            # Content quality recommendations
            if stats['content_quality'].get('empty_content_count', 0) > 0:
                recommendations.append(f"📝 Remove {stats['content_quality']['empty_content_count']} empty examples from {component}")
            
            if stats['content_quality'].get('legal_content_percentage', 0) < 70:
                recommendations.append(f"⚖️ Improve legal terminology coverage in {component} (current: {stats['content_quality'].get('legal_content_percentage', 0):.1f}%)")
            
            # Diversity recommendations
            variety_score = stats['diversity_metrics'].get('input_variety_score', 0)
            if variety_score < 0.7:
                recommendations.append(f"🎯 Increase input diversity for {component} (current variety: {variety_score:.1%})")
        
        # Overall recommendations
        if overall_stats['total_examples'] < 10000:
            recommendations.append("📊 Consider expanding dataset - current size is suitable for initial training but more data will improve performance")
        
        if not recommendations:
            recommendations.append("✅ Data quality looks good! Ready for model training.")
        
        return recommendations
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive quality report"""
        stats = self.check_all_components()
        
        report_lines = [
            "=" * 80,
            "🔍 AILES LEGAL AI - DATA QUALITY REPORT",
            "=" * 80,
            f"📊 Total Examples: {stats['total_examples']:,}",
            f"📁 Components Checked: {len(stats['components'])}",
            ""
        ]
        
        # Component-by-component analysis
        for component, component_stats in stats['components'].items():
            report_lines.extend([
                f"📋 {component.upper()} COMPONENT:",
                f"   📄 Examples: {component_stats['example_count']:,}",
                f"   💾 File Size: {component_stats['file_size_mb']:.1f} MB",
                f"   ✅ Valid JSON: {component_stats['valid_json_count']}/{component_stats['example_count']}",
                f"   ⚠️ Schema Issues: {len(component_stats['schema_violations'])}",
                f"   📝 Avg Input Length: {component_stats['content_quality'].get('avg_input_length', 0):.0f} chars",
                f"   📤 Avg Output Length: {component_stats['content_quality'].get('avg_output_length', 0):.0f} chars",
                f"   🎯 Input Variety: {component_stats['diversity_metrics'].get('input_variety_score', 0):.1%}",
                f"   ⚖️ Legal Content: {component_stats['content_quality'].get('legal_content_percentage', 0):.1f}%",
                ""
            ])
            
            # Show distribution data
            if component == 'chatbot' and component_stats['diversity_metrics'].get('qualification_distribution'):
                report_lines.append("   🎯 Qualification Distribution:")
                for qual, count in component_stats['diversity_metrics']['qualification_distribution'].items():
                    report_lines.append(f"      {qual}: {count}")
                report_lines.append("")
        
        # Issues and recommendations
        if stats['quality_issues']:
            report_lines.extend([
                "⚠️ QUALITY ISSUES:",
                *[f"   • {issue}" for issue in stats['quality_issues']],
                ""
            ])
        
        report_lines.extend([
            "💡 RECOMMENDATIONS:",
            *[f"   • {rec}" for rec in stats['recommendations']],
            "",
            "=" * 80,
            f"📅 Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"📄 Quality report saved to: {output_file}")
        
        return report

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check AILES training data quality")
    parser.add_argument("--data_dir", default="data/processed", help="Directory containing training data")
    parser.add_argument("--output", help="Output file for quality report")
    parser.add_argument("--component", choices=['chatbot', 'predictor', 'explainer'], help="Check specific component only")
    
    args = parser.parse_args()
    
    checker = DataQualityChecker(args.data_dir)
    
    if args.component:
        # Check single component
        file_path = Path(args.data_dir) / f"{args.component}_training_data.jsonl"
        if file_path.exists():
            stats = checker.check_component_quality(args.component, file_path)
            print(f"\n📊 {args.component.upper()} Quality Check:")
            print(f"   Examples: {stats['example_count']:,}")
            print(f"   Valid JSON: {stats['valid_json_count']}")
            print(f"   Schema Issues: {len(stats['schema_violations'])}")
            print(f"   Input Variety: {stats['diversity_metrics'].get('input_variety_score', 0):.1%}")
        else:
            print(f"❌ File not found: {file_path}")
    else:
        # Generate full report
        report = checker.generate_report(args.output)
        print(report)

if __name__ == "__main__":
    main()