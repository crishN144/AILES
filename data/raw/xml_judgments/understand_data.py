#!/usr/bin/env python3
"""
Complete Dataset Analyzer for AILES Legal AI
Analyzes ALL 9,834 XML files to understand the dataset composition
"""

import xml.etree.ElementTree as ET
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict, Counter
import json

class DatasetAnalyzer:
    def __init__(self, xml_directory: str):
        self.xml_dir = Path(xml_directory)
        
        # Comprehensive case type patterns
        self.case_patterns = {
            'financial_remedy': [
                'financial remedy', 'financial relief', 'financial provision', 
                'ancillary relief', 'financial order', 'lump sum', 'periodical payments',
                'maintenance', 'financial settlement', 'matrimonial finance'
            ],
            'child_arrangements': [
                'child arrangements', 'contact', 'residence', 'custody', 
                'parental responsibility', 'specific issue', 'prohibited steps',
                'children act', 'section 8', 'child welfare'
            ],
            'divorce_dissolution': [
                'divorce', 'dissolution', 'matrimonial causes', 'decree nisi',
                'decree absolute', 'judicial separation', 'nullity'
            ],
            'domestic_violence': [
                'domestic violence', 'domestic abuse', 'non-molestation',
                'occupation order', 'forced marriage', 'harassment',
                'family law act 1996'
            ],
            'adoption_fostering': [
                'adoption', 'adopt', 'foster', 'fostering', 'placement order',
                'special guardianship', 'adoption act', 'freeing for adoption'
            ],
            'care_proceedings': [
                'care order', 'care proceedings', 'emergency protection',
                'supervision order', 'interim care', 'threshold criteria',
                'significant harm', 'care plan'
            ],
            'international_family': [
                'international', 'hague convention', 'brussels', 'jurisdiction',
                'cross-border', 'foreign', 'abduction', 'recognition'
            ],
            'cohabitation': [
                'cohabitation', 'unmarried', 'civil partnership', 
                'schedule 1', 'trusts of land', 'beneficial interest'
            ],
            'mental_capacity': [
                'mental capacity', 'court of protection', 'best interests',
                'mental health', 'capacity act', 'deprivation of liberty'
            ],
            'inheritance_family': [
                'inheritance act', 'family provision', 'probate',
                'estate', 'will', 'intestacy', 'reasonable provision'
            ]
        }
        
        # Court classifications
        self.court_types = {
            'EWHC': 'High Court',
            'EWFC': 'Family Court', 
            'EWCOP': 'Court of Protection',
            'EWCA': 'Court of Appeal',
            'UKSC': 'Supreme Court'
        }
        
        # Financial indicators
        self.financial_indicators = [
            '£', 'income', 'salary', 'property', 'mortgage', 'assets',
            'pension', 'maintenance', 'capital', 'equity', 'valuati',
            'form e', 'financial statement'
        ]

    def analyze_complete_dataset(self) -> Dict[str, Any]:
        """Analyze the complete dataset of 9,834 files"""
        print("🔍 COMPLETE DATASET ANALYSIS STARTING...")
        print("="*60)
        
        xml_files = self.find_all_xml_files()
        total_files = len(xml_files)
        
        print(f"📁 Found {total_files:,} XML files")
        print(f"📊 Analyzing all files... (this may take 5-10 minutes)")
        
        results = {
            'total_files': total_files,
            'valid_files': 0,
            'invalid_files': 0,
            'case_types': defaultdict(int),
            'courts': defaultdict(int),
            'years': defaultdict(int),
            'file_sizes': [],
            'financial_data_cases': 0,
            'complexity_levels': defaultdict(int),
            'sample_cases_by_type': defaultdict(list),
            'errors': [],
            'year_range': {'min': 3000, 'max': 1900},
            'top_keywords': Counter()
        }
        
        # Process all files
        for i, xml_file in enumerate(xml_files):
            if (i + 1) % 500 == 0:
                print(f"  📈 Progress: {i+1:,}/{total_files:,} ({(i+1)/total_files*100:.1f}%)")
            
            try:
                file_info = self._analyze_single_file(xml_file)
                if file_info:
                    results['valid_files'] += 1
                    
                    # Aggregate data
                    case_type = file_info['primary_type']
                    results['case_types'][case_type] += 1
                    results['courts'][file_info['court']] += 1
                    results['years'][file_info['year']] += 1
                    results['file_sizes'].append(file_info['size_kb'])
                    results['complexity_levels'][file_info['complexity']] += 1
                    
                    # Track year range
                    year = file_info['year']
                    if year < results['year_range']['min']:
                        results['year_range']['min'] = year
                    if year > results['year_range']['max']:
                        results['year_range']['max'] = year
                    
                    # Financial data tracking
                    if file_info['has_financial']:
                        results['financial_data_cases'] += 1
                    
                    # Keep sample cases for each type (max 5 per type)
                    if len(results['sample_cases_by_type'][case_type]) < 5:
                        results['sample_cases_by_type'][case_type].append({
                            'filename': xml_file.name,
                            'citation': file_info['citation'],
                            'year': file_info['year'],
                            'court': file_info['court']
                        })
                    
                    # Track keywords
                    for keyword in file_info['keywords']:
                        results['top_keywords'][keyword] += 1
                else:
                    results['invalid_files'] += 1
                    
            except Exception as e:
                results['invalid_files'] += 1
                results['errors'].append(f"{xml_file.name}: {str(e)[:100]}")
        
        # Calculate statistics
        results['validity_rate'] = (results['valid_files'] / total_files) * 100
        results['financial_data_rate'] = (results['financial_data_cases'] / results['valid_files']) * 100 if results['valid_files'] > 0 else 0
        results['avg_file_size'] = sum(results['file_sizes']) / len(results['file_sizes']) if results['file_sizes'] else 0
        
        return results
    
    def find_all_xml_files(self) -> List[Path]:
        """Find all XML files recursively"""
        xml_files = []
        
        # Look in current directory and subdirectories
        for pattern in ["*.xml", "**/*.xml"]:
            found = list(self.xml_dir.glob(pattern))
            xml_files.extend(found)
        
        # Remove duplicates and sort
        xml_files = sorted(list(set(xml_files)))
        return xml_files
    
    def _analyze_single_file(self, xml_file: Path) -> Dict[str, Any]:
        """Analyze a single XML file comprehensively"""
        try:
            # Parse XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract all text content
            all_text = self._extract_all_text(root).lower()
            
            # Basic file info
            file_size_kb = xml_file.stat().st_size // 1024
            
            # Extract metadata
            citation = self._extract_citation(root, all_text)
            court = self._extract_court(citation, all_text)
            year = self._extract_year(citation, xml_file.name)
            
            # Classify case type
            primary_type, all_types = self._classify_case_comprehensive(all_text)
            
            # Check for financial data
            has_financial = self._check_financial_content(all_text)
            
            # Assess complexity
            complexity = self._assess_complexity(all_text, all_types)
            
            # Extract key terms
            keywords = self._extract_key_terms(all_text)
            
            return {
                'filename': xml_file.name,
                'citation': citation,
                'court': court,
                'year': year,
                'primary_type': primary_type,
                'all_types': all_types,
                'has_financial': has_financial,
                'complexity': complexity,
                'size_kb': file_size_kb,
                'text_length': len(all_text),
                'keywords': keywords
            }
            
        except Exception as e:
            return None
    
    def _extract_all_text(self, element) -> str:
        """Extract all text from XML"""
        text_parts = []
        
        if element.text:
            text_parts.append(element.text.strip())
        
        for child in element:
            child_text = self._extract_all_text(child)
            if child_text:
                text_parts.append(child_text)
            if child.tail:
                text_parts.append(child.tail.strip())
        
        return ' '.join(text_parts)
    
    def _extract_citation(self, root, text: str) -> str:
        """Extract case citation"""
        # Try XML attributes first
        citation_patterns = [
            './/FRBRthis',
            './/cite', 
            './/citation'
        ]
        
        for pattern in citation_patterns:
            elem = root.find(pattern)
            if elem is not None:
                value = elem.get('value') or elem.text
                if value and '[' in value:
                    return value
        
        # Fallback to text extraction
        citation_match = re.search(r'\[20\d{2}\] [A-Z]+ \d+', text)
        if citation_match:
            return citation_match.group()
        
        return "Unknown"
    
    def _extract_court(self, citation: str, text: str) -> str:
        """Extract court type"""
        if 'EWHC' in citation and 'Fam' in citation:
            return 'EWHC-Family'
        elif 'EWFC' in citation:
            return 'Family Court'
        elif 'EWCOP' in citation:
            return 'Court of Protection'
        elif 'EWCA' in citation:
            return 'Court of Appeal'
        elif 'UKSC' in citation:
            return 'Supreme Court'
        else:
            return 'Other'
    
    def _extract_year(self, citation: str, filename: str) -> int:
        """Extract year from citation or filename"""
        # Try citation first
        year_match = re.search(r'\[(\d{4})\]', citation)
        if year_match:
            return int(year_match.group(1))
        
        # Try filename
        year_match = re.search(r'\[(\d{4})\]', filename)
        if year_match:
            return int(year_match.group(1))
        
        return 2010  # Default
    
    def _classify_case_comprehensive(self, text: str) -> tuple:
        """Comprehensive case type classification"""
        type_scores = {}
        
        # Score each case type
        for case_type, patterns in self.case_patterns.items():
            score = 0
            for pattern in patterns:
                score += text.count(pattern)
            if score > 0:
                type_scores[case_type] = score
        
        if not type_scores:
            return 'unclassified', []
        
        # Primary type is highest scoring
        primary_type = max(type_scores, key=type_scores.get)
        
        # All types with significant presence (score >= 2)
        all_types = [t for t, s in type_scores.items() if s >= 2]
        
        return primary_type, all_types
    
    def _check_financial_content(self, text: str) -> bool:
        """Check for substantial financial content"""
        financial_score = sum(text.count(indicator) for indicator in self.financial_indicators)
        return financial_score >= 5  # Threshold for "substantial" financial content
    
    def _assess_complexity(self, text: str, case_types: List[str]) -> str:
        """Assess case complexity"""
        complexity_score = 0
        
        # Multiple case types = more complex
        complexity_score += len(case_types)
        
        # Specific complexity indicators
        complex_indicators = [
            'appeal', 'cross-appeal', 'judicial review',
            'international', 'jurisdiction', 'forum non conveniens',
            'expert evidence', 'psychiatric', 'forensic',
            'business assets', 'pension sharing', 'trust',
            'multiple parties', 'intervener', 'joinder'
        ]
        
        for indicator in complex_indicators:
            if indicator in text:
                complexity_score += 1
        
        if complexity_score >= 5:
            return 'high'
        elif complexity_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key legal terms for understanding dataset"""
        key_terms = []
        
        # Important family law terms
        important_terms = [
            'children act', 'matrimonial causes act', 'family law act',
            'domestic violence', 'forced marriage', 'female genital mutilation',
            'hague convention', 'brussels regulation', 'maintenance regulation',
            'pension sharing', 'lump sum', 'periodical payments',
            'contact order', 'residence order', 'specific issue order',
            'care order', 'supervision order', 'emergency protection',
            'adoption order', 'placement order', 'special guardianship'
        ]
        
        for term in important_terms:
            if term in text:
                key_terms.append(term)
        
        return key_terms[:10]  # Top 10 terms per case
    
    def create_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Create a detailed analysis report"""
        report = f"""
# AILES Legal AI Dataset Analysis Report
Generated: {self._get_timestamp()}

## 📊 DATASET OVERVIEW
- **Total Files**: {results['total_files']:,}
- **Valid Files**: {results['valid_files']:,} ({results['validity_rate']:.1f}%)
- **Invalid Files**: {results['invalid_files']:,}
- **Average File Size**: {results['avg_file_size']:.1f} KB
- **Year Range**: {results['year_range']['min']} - {results['year_range']['max']}

## 📋 CASE TYPE DISTRIBUTION
"""
        
        # Sort case types by frequency
        sorted_types = sorted(results['case_types'].items(), key=lambda x: x[1], reverse=True)
        for case_type, count in sorted_types:
            percentage = (count / results['valid_files']) * 100
            report += f"- **{case_type.replace('_', ' ').title()}**: {count:,} cases ({percentage:.1f}%)\n"
        
        report += f"""
## 🏛️ COURT DISTRIBUTION
"""
        sorted_courts = sorted(results['courts'].items(), key=lambda x: x[1], reverse=True)
        for court, count in sorted_courts:
            percentage = (count / results['valid_files']) * 100
            report += f"- **{court}**: {count:,} cases ({percentage:.1f}%)\n"
        
        report += f"""
## 📅 TEMPORAL DISTRIBUTION
"""
        # Group years into decades
        decades = defaultdict(int)
        for year, count in results['years'].items():
            decade = (year // 10) * 10
            decades[decade] += count
        
        for decade in sorted(decades.keys()):
            report += f"- **{decade}s**: {decades[decade]:,} cases\n"
        
        report += f"""
## 💰 FINANCIAL DATA ANALYSIS
- **Cases with Financial Data**: {results['financial_data_cases']:,} ({results['financial_data_rate']:.1f}%)
- **Suitable for Predictor Training**: {results['financial_data_cases']:,} cases

## ⚖️ COMPLEXITY DISTRIBUTION
"""
        for complexity, count in sorted(results['complexity_levels'].items()):
            percentage = (count / results['valid_files']) * 100
            report += f"- **{complexity.title()} Complexity**: {count:,} cases ({percentage:.1f}%)\n"
        
        report += f"""
## 🎯 SAMPLING RECOMMENDATIONS

### For Quick Start (Emergency - 100 cases):
- Financial Remedy: {min(30, results['case_types'].get('financial_remedy', 0))} cases
- Child Arrangements: {min(25, results['case_types'].get('child_arrangements', 0))} cases  
- Divorce: {min(20, results['case_types'].get('divorce_dissolution', 0))} cases
- Domestic Violence: {min(10, results['case_types'].get('domestic_violence', 0))} cases
- Other types: {15} cases

### For Production (300 cases):
- Balanced representation across all case types
- Include high/medium/low complexity cases
- Ensure good financial data coverage

### For Full Training (1000+ cases):
- Representative sample maintaining proportions
- Include edge cases and appeals
- Full temporal coverage

## 🚨 QUALITY ISSUES
- **Error Rate**: {(results['invalid_files'] / results['total_files']) * 100:.1f}%
- **Common Errors**: {len(set(error.split(':')[1] if ':' in error else error for error in results['errors'][:10]))} unique error types

## 🔍 TOP LEGAL TERMS FOUND
"""
        for term, count in results['top_keywords'].most_common(15):
            report += f"- **{term}**: {count:,} cases\n"
        
        return report
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def create_smart_sample(self, results: Dict[str, Any], target_size: int = 200) -> List[str]:
        """Create a smart sample based on analysis results"""
        print(f"\n🎯 Creating smart sample of {target_size} cases...")
        
        # Calculate proportional sampling
        total_valid = results['valid_files']
        sample_plan = {}
        
        for case_type, count in results['case_types'].items():
            proportion = count / total_valid
            target_for_type = max(1, int(target_size * proportion))
            sample_plan[case_type] = min(target_for_type, count)
        
        print("📋 Sampling plan:")
        for case_type, target in sample_plan.items():
            print(f"  {case_type.replace('_', ' ').title()}: {target} cases")
        
        # Get files for each type from the samples we collected
        selected_files = []
        for case_type, target_count in sample_plan.items():
            available_samples = results['sample_cases_by_type'].get(case_type, [])
            for sample in available_samples[:target_count]:
                selected_files.append(sample['filename'])
        
        # If we need more files, add from the most common types
        while len(selected_files) < target_size:
            # Add more from largest categories
            largest_type = max(results['case_types'], key=results['case_types'].get)
            available = results['sample_cases_by_type'].get(largest_type, [])
            
            for sample in available:
                if sample['filename'] not in selected_files:
                    selected_files.append(sample['filename'])
                    break
            else:
                break  # No more files available
        
        return selected_files[:target_size]
    
    def save_analysis_results(self, results: Dict[str, Any], selected_files: List[str]):
        """Save all analysis results"""
        # Save detailed JSON results
        with open('dataset_analysis.json', 'w') as f:
            # Convert defaultdicts to regular dicts for JSON serialization
            json_results = {
                k: dict(v) if isinstance(v, defaultdict) else v 
                for k, v in results.items() 
                if k not in ['top_keywords']  # Skip Counter objects
            }
            json_results['top_keywords'] = dict(results['top_keywords'].most_common(50))
            json.dump(json_results, f, indent=2)
        
        # Save human-readable report
        report = self.create_comprehensive_report(results)
        with open('dataset_analysis_report.md', 'w') as f:
            f.write(report)
        
        # Save selected files list
        with open('selected_training_files.txt', 'w') as f:
            for filename in selected_files:
                f.write(filename + '\n')
        
        print("💾 Analysis results saved:")
        print("  📊 dataset_analysis.json - Raw data")
        print("  📝 dataset_analysis_report.md - Human readable report")  
        print("  📁 selected_training_files.txt - Files for training")

def main():
    """Main execution function"""
    print("🚀 AILES Legal AI - Complete Dataset Analyzer")
    print("="*60)
    
    # Configure your XML directory path
    xml_directory = "."
    
    # For direct analysis from Downloads (if you haven't copied files yet)
    downloads_path = Path.home() / "Downloads" / "National-Archives - Family extract"
    if downloads_path.exists() and not Path(xml_directory).exists():
        print(f"📁 Using files directly from Downloads folder")
        xml_directory = str(downloads_path)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(xml_directory)
    
    # Run complete analysis
    results = analyzer.analyze_complete_dataset()
    
    # Print summary
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📊 Processed {results['total_files']:,} files")
    print(f"✅ Valid: {results['valid_files']:,} ({results['validity_rate']:.1f}%)")
    print(f"💰 Financial data: {results['financial_data_cases']:,} cases")
    
    # Create smart sample
    selected_files = analyzer.create_smart_sample(results, target_size=200)
    
    # Save everything
    analyzer.save_analysis_results(results, selected_files)
    
    print(f"\n🎉 COMPLETE DATASET ANALYSIS FINISHED!")
    print(f"📈 Check 'dataset_analysis_report.md' for full details")
    print(f"🎯 Ready to train with {len(selected_files)} selected cases")

if __name__ == "__main__":
    main()