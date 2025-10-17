#!/usr/bin/env python3
"""
Enhanced XML Legal Judgment Parser for AILES Training Data Generation
Adapted for comprehensive family law dataset with inheritance, mental capacity, international cases
"""

import xml.etree.ElementTree as ET
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import random

class EnhancedLegalXMLProcessor:
    def __init__(self):
        # Define XML namespaces for your data
        self.namespaces = {
            'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0',
            'html': 'http://www.w3.org/1999/xhtml',
            'uk': 'https://caselaw.nationalarchives.gov.uk/akn'
        }
        
        # Enhanced financial patterns based on your dataset
        self.financial_patterns = {
            'inheritance_value': r'estate(?:\s+valued?\s+at)?\s*£?\d{1,3}(?:,\d{3})*',
            'property_value': r'(?:property|house|home|real estate)(?:\s+valued?\s+at)?\s*£?\d{1,3}(?:,\d{3})*',
            'income': r'£?\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:per|annually|monthly|weekly|year|annum|p\.a\.))?',
            'lump_sum': r'lump\s+sum\s*(?:of)?\s*£?\d{1,3}(?:,\d{3})*',
            'periodical_payments': r'periodical\s+payments?\s*(?:of)?\s*£?\d{1,3}(?:,\d{3})*',
            'maintenance': r'(?:maintenance|support|alimony)\s*(?:of|payments?)?\s*£?\d{1,3}(?:,\d{3})*',
            'pension': r'pension(?:\s+(?:sharing|credit|attachment))?\s*(?:of)?\s*£?\d{1,3}(?:,\d{3})*',
            'costs': r'(?:costs|fees?|expenses?)\s*(?:of)?\s*£?\d{1,3}(?:,\d{3})*'
        }
        
        # Comprehensive case type classification based on your actual dataset
        self.case_type_keywords = {
            'inheritance_family': [
                'inheritance act', 'family provision', 'reasonable provision', 
                'deceased estate', 'beneficiary', 'intestacy', 'will contest',
                'dependency claim', 'family inheritance', 'estate claim'
            ],
            'child_arrangements': [
                'child arrangements order', 'residence order', 'contact order',
                'specific issue order', 'prohibited steps', 'section 8',
                'custody', 'parental responsibility', 'child welfare'
            ],
            'mental_capacity': [
                'mental capacity act', 'court of protection', 'best interests',
                'capacity assessment', 'deputyship', 'lasting power of attorney',
                'mental health', 'incapacity', 'welfare decision'
            ],
            'international_family': [
                'hague convention', 'international child abduction', 'jurisdiction',
                'foreign divorce', 'brussels regulation', 'cross border',
                'international element', 'foreign court', 'treaty'
            ],
            'adoption_fostering': [
                'adoption order', 'placement order', 'adoption agency',
                'foster care', 'special guardianship', 'care plan',
                'adoption panel', 'permanence'
            ],
            'care_proceedings': [
                'care order', 'supervision order', 'emergency protection',
                'children act 1989', 'threshold criteria', 'local authority',
                'interim care order', 'child protection'
            ],
            'financial_remedy': [
                'financial remedy', 'matrimonial causes act', 'ancillary relief',
                'financial provision', 'property adjustment', 'clean break',
                'spousal maintenance', 'financial settlement'
            ],
            'divorce_dissolution': [
                'divorce', 'dissolution', 'decree nisi', 'decree absolute',
                'irretrievable breakdown', 'matrimonial', 'civil partnership'
            ],
            'domestic_violence': [
                'domestic violence', 'non-molestation order', 'occupation order',
                'family law act 1996', 'harassment', 'domestic abuse',
                'protective order', 'restraining order'
            ],
            'cohabitation': [
                'cohabitation', 'unmarried couples', 'civil partnership',
                'schedule 1', 'property disputes', 'trust'
            ]
        }
        
        # Enhanced user scenario templates based on your dataset categories
        self.scenario_templates = {
            'inheritance_family': [
                "My parent died and I believe I should have been provided for in the will.",
                "There's a family dispute over my father's estate - I was left nothing but I was dependent on him.",
                "My spouse died and the family are contesting their will. I need help understanding my rights.",
                "I was promised inheritance but it's not in the will. Can I claim under the Inheritance Act?"
            ],
            'child_arrangements': [
                "My ex and I can't agree on where our children should live after separation.",
                "I want to change the custody arrangements - my ex won't let me see the kids.",
                "We need help sorting out child arrangements after our relationship ended.",
                "My ex wants to move abroad with our children and I don't agree."
            ],
            'mental_capacity': [
                "My elderly parent lacks mental capacity and we need help with their financial affairs.",
                "There are concerns about my relative's mental capacity to make decisions.",
                "I need to apply to be someone's deputy as they can't manage their own affairs.",
                "We need help making decisions for someone who lacks mental capacity."
            ],
            'international_family': [
                "My ex has taken our children abroad without my permission.",
                "We're from different countries and getting divorced - which law applies?",
                "My children are being held in another country by my ex-partner.",
                "We need help with an international custody dispute."
            ],
            'adoption_fostering': [
                "We want to adopt a child and need guidance on the legal process.",
                "I'm considering placing my child for adoption - what are my rights?",
                "We're foster parents wanting to adopt our foster child.",
                "There are problems with our adoption proceedings."
            ],
            'care_proceedings': [
                "Social services are involved with my family - what does this mean?",
                "The local authority wants to take my children into care.",
                "I'm worried about my child's safety with their other parent.",
                "We're facing care proceedings and need help understanding our rights."
            ],
            'financial_remedy': [
                "We're getting divorced and need help dividing our finances and property.",
                "My spouse earns much more than me - what financial support can I get?",
                "We can't agree on how to split our assets in the divorce.",
                "I need help understanding what I'm entitled to financially after divorce."
            ],
            'divorce_dissolution': [
                "I want to get divorced but don't know where to start.",
                "My spouse wants a divorce but I don't - what are my rights?",
                "We're separated and considering making it official with divorce.",
                "How long does the divorce process take and what's involved?"
            ],
            'domestic_violence': [
                "My partner is abusive and I need protection for me and my children.",
                "I need to get my ex removed from our home for safety reasons.",
                "I have a restraining order but my ex keeps contacting me.",
                "I'm experiencing domestic abuse and need urgent legal help."
            ],
            'cohabitation': [
                "We're not married but have been together for years - what are my rights?",
                "My unmarried partner and I own a house together and we're splitting up.",
                "We have children together but were never married - what happens now?",
                "I need help with property rights as an unmarried partner."
            ]
        }

        # Legal complexity indicators based on your dataset
        self.complexity_keywords = {
            'high_complexity': [
                'international element', 'hague convention', 'jurisdiction', 'foreign court',
                'mental capacity', 'court of protection', 'complex financial', 'trust',
                'inheritance act', 'family provision', 'contested', 'appeal',
                'care proceedings', 'threshold criteria', 'expert evidence'
            ],
            'medium_complexity': [
                'child arrangements', 'financial remedy', 'property adjustment',
                'adoption', 'supervision order', 'contact dispute', 'maintenance'
            ],
            'low_complexity': [
                'consent order', 'agreed arrangements', 'straightforward', 'uncontested'
            ]
        }

    def classify_case_type(self, text: str) -> str:
        """Classify case type based on content analysis"""
        text_lower = text.lower()
        
        # Count keyword matches for each case type
        type_scores = {}
        for case_type, keywords in self.case_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                type_scores[case_type] = score
        
        # Return the type with highest score, or 'unclassified' if no matches
        if type_scores:
            return max(type_scores.keys(), key=lambda x: type_scores[x])
        return 'unclassified'

    def assess_complexity(self, text: str) -> str:
        """Assess case complexity based on indicators"""
        text_lower = text.lower()
        
        complexity_scores = {
            'high': sum(1 for keyword in self.complexity_keywords['high_complexity'] if keyword in text_lower),
            'medium': sum(1 for keyword in self.complexity_keywords['medium_complexity'] if keyword in text_lower),
            'low': sum(1 for keyword in self.complexity_keywords['low_complexity'] if keyword in text_lower)
        }
        
        # Determine complexity based on highest score
        max_complexity = max(complexity_scores.keys(), key=lambda x: complexity_scores[x])
        
        # If no clear indicators, default to medium
        if complexity_scores[max_complexity] == 0:
            return 'medium'
        
        return max_complexity

    def extract_judgment_data(self, xml_file: Path) -> Optional[Dict[str, Any]]:
        """Extract structured data from XML judgment file"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            judgment_data = {
                'file_name': xml_file.name,
                'case_citation': '',
                'court': '',
                'judge': '',
                'date': '',
                'parties': {},
                'case_facts': '',
                'legal_reasoning': '',
                'decision': '',
                'financial_info': {},
                'case_type': '',
                'complexity': '',
                'main_issues': []
            }
            
            # Extract metadata
            self._extract_metadata(root, judgment_data)
            
            # Extract judgment content
            self._extract_content(root, judgment_data)
            
            # Classify case type
            full_text = (judgment_data['case_facts'] + ' ' + 
                        judgment_data['legal_reasoning'] + ' ' + 
                        judgment_data['decision'])
            
            judgment_data['case_type'] = self.classify_case_type(full_text)
            judgment_data['complexity'] = self.assess_complexity(full_text)
            
            # Extract financial information
            self._extract_financial_data(judgment_data)
            
            # Identify main issues
            self._identify_main_issues(judgment_data)
            
            return judgment_data
            
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return None

    def _extract_metadata(self, root, judgment_data):
        """Extract case metadata"""
        # Try multiple XPath expressions for flexibility
        citation_paths = [
            './/{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}FRBRthis',
            './/FRBRthis',
            './/*[@value]'
        ]
        
        for path in citation_paths:
            elements = root.findall(path)
            for elem in elements:
                if elem.get('value'):
                    judgment_data['case_citation'] = elem.get('value', '')
                    break
            if judgment_data['case_citation']:
                break
        
        # Extract judge information
        judge_paths = [
            './/{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}TLCPerson[@as="judge"]',
            './/TLCPerson[@as="judge"]',
            './/*[@as="judge"]'
        ]
        
        for path in judge_paths:
            judges = root.findall(path)
            if judges:
                judgment_data['judge'] = judges[0].get('showAs', '')
                break
        
        # Extract date
        date_paths = [
            './/{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}FRBRdate',
            './/FRBRdate',
            './/*[@date]'
        ]
        
        for path in date_paths:
            date_elem = root.find(path)
            if date_elem is not None:
                judgment_data['date'] = date_elem.get('date', '')
                break

    def _extract_content(self, root, judgment_data):
        """Extract main judgment content with flexible parsing"""
        # Try to find content in various structures
        content_paths = [
            './/{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}level',
            './/level',
            './/{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}p',
            './/p',
            './/{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}div',
            './/div'
        ]
        
        all_content = []
        
        for path in content_paths:
            elements = root.findall(path)
            if elements:
                for elem in elements:
                    text = self._extract_text_from_element(elem)
                    if text and len(text.strip()) > 50:  # Only substantial content
                        all_content.append(text)
                break  # Use first successful path
        
        # If no structured content found, get all text
        if not all_content:
            all_text = self._extract_text_from_element(root)
            all_content = [all_text]
        
        # Categorize content based on headings and context
        case_facts = []
        reasoning = []
        decision = []
        
        for content in all_content:
            content_lower = content.lower()
            
            # Classify content sections
            if any(word in content_lower for word in ['facts', 'background', 'circumstances', 'history']):
                case_facts.append(content)
            elif any(word in content_lower for word in ['reasoning', 'analysis', 'discussion', 'consideration', 'law']):
                reasoning.append(content)
            elif any(word in content_lower for word in ['decision', 'conclusion', 'order', 'declaration', 'judgment']):
                decision.append(content)
            else:
                # Default to reasoning for substantial content
                if len(content.strip()) > 200:
                    reasoning.append(content)
        
        # Combine sections
        judgment_data['case_facts'] = '\n\n'.join(case_facts) if case_facts else all_content[0] if all_content else ''
        judgment_data['legal_reasoning'] = '\n\n'.join(reasoning) if reasoning else ''
        judgment_data['decision'] = '\n\n'.join(decision) if decision else ''

    def _extract_text_from_element(self, element) -> str:
        """Extract all text content from XML element recursively"""
        text_parts = []
        
        if element.text:
            text_parts.append(element.text.strip())
        
        for child in element:
            child_text = self._extract_text_from_element(child)
            if child_text:
                text_parts.append(child_text)
            if child.tail:
                text_parts.append(child.tail.strip())
        
        return ' '.join(text_parts)

    def _extract_financial_data(self, judgment_data):
        """Extract and structure financial information based on case type"""
        full_text = (
            judgment_data['case_facts'] + ' ' + 
            judgment_data['legal_reasoning'] + ' ' + 
            judgment_data['decision']
        ).lower()
        
        financial_info = {
            'has_financial_data': False,
            'financial_amounts': [],
            'financial_summary': '',
            'case_specific_financial': {}
        }
        
        case_type = judgment_data['case_type']
        
        # Extract different financial patterns based on case type
        if case_type == 'inheritance_family':
            estate_matches = re.findall(self.financial_patterns['inheritance_value'], full_text)
            if estate_matches:
                financial_info['case_specific_financial']['estate_value'] = estate_matches
                financial_info['has_financial_data'] = True
        
        elif case_type in ['financial_remedy', 'divorce_dissolution']:
            lump_sum_matches = re.findall(self.financial_patterns['lump_sum'], full_text)
            periodical_matches = re.findall(self.financial_patterns['periodical_payments'], full_text)
            if lump_sum_matches or periodical_matches:
                financial_info['case_specific_financial']['lump_sum'] = lump_sum_matches
                financial_info['case_specific_financial']['periodical_payments'] = periodical_matches
                financial_info['has_financial_data'] = True
        
        elif case_type == 'child_arrangements':
            maintenance_matches = re.findall(self.financial_patterns['maintenance'], full_text)
            if maintenance_matches:
                financial_info['case_specific_financial']['child_maintenance'] = maintenance_matches
                financial_info['has_financial_data'] = True
        
        # Extract general financial patterns for all cases
        property_matches = re.findall(self.financial_patterns['property_value'], full_text)
        income_matches = re.findall(self.financial_patterns['income'], full_text)
        
        if property_matches:
            financial_info['case_specific_financial']['property'] = property_matches[:3]
            financial_info['has_financial_data'] = True
        
        if income_matches:
            financial_info['case_specific_financial']['income'] = income_matches[:3]
            financial_info['has_financial_data'] = True
        
        # Create financial summary
        summary_parts = []
        for key, values in financial_info['case_specific_financial'].items():
            if values:
                summary_parts.append(f"{key.replace('_', ' ').title()}: {', '.join(values[:2])}")
        
        financial_info['financial_summary'] = '; '.join(summary_parts)
        
        judgment_data['financial_info'] = financial_info

    def _identify_main_issues(self, judgment_data):
        """Identify main legal issues based on case type and content"""
        case_type = judgment_data['case_type']
        issues = []
        
        # Map case types to main issues
        issue_mapping = {
            'inheritance_family': ['Family provision', 'Estate distribution', 'Inheritance Act claim'],
            'child_arrangements': ['Child custody/arrangements', 'Parental responsibility', 'Child welfare'],
            'mental_capacity': ['Mental capacity assessment', 'Best interests decision', 'Deputyship'],
            'international_family': ['International jurisdiction', 'Child abduction', 'Cross-border disputes'],
            'adoption_fostering': ['Adoption procedure', 'Child placement', 'Parental consent'],
            'care_proceedings': ['Child protection', 'Care orders', 'Local authority intervention'],
            'financial_remedy': ['Financial provision', 'Property division', 'Spousal maintenance'],
            'divorce_dissolution': ['Divorce procedure', 'Matrimonial breakdown', 'Legal separation'],
            'domestic_violence': ['Domestic abuse protection', 'Non-molestation orders', 'Personal safety'],
            'cohabitation': ['Unmarried rights', 'Property disputes', 'Cohabitation claims']
        }
        
        if case_type in issue_mapping:
            issues = issue_mapping[case_type]
        else:
            issues = ['Legal dispute resolution']
        
        judgment_data['main_issues'] = issues

    def create_training_examples(self, judgment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create training examples for all three AI components based on case type"""
        examples = []
        
        # Create examples based on case type and available data
        if judgment_data['case_type'] != 'unclassified':
            # Always create chatbot examples
            examples.extend(self._create_chatbot_examples(judgment_data))
            
            # Create predictor examples if financial data available
            if judgment_data['financial_info']['has_financial_data']:
                examples.extend(self._create_predictor_examples(judgment_data))
            
            # Create explainer examples if substantial legal reasoning
            if len(judgment_data['legal_reasoning']) > 300:
                examples.extend(self._create_explainer_examples(judgment_data))
        
        return examples

    def _create_chatbot_examples(self, judgment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chatbot training examples based on case type"""
        examples = []
        case_type = judgment_data['case_type']
        complexity = judgment_data['complexity']
        
        # Determine qualification based on case type and complexity
        if case_type in ['inheritance_family', 'mental_capacity', 'international_family', 'care_proceedings']:
            qualification = "QUALIFY_CASE"  # Complex cases need full assessment
        elif case_type in ['financial_remedy', 'adoption_fostering', 'domestic_violence']:
            qualification = "QUALIFY_ADVISOR"  # Need professional advice
        elif case_type in ['child_arrangements', 'divorce_dissolution', 'cohabitation']:
            if complexity == 'high':
                qualification = "QUALIFY_CASE"
            else:
                qualification = "QUALIFY_ADVISOR"
        else:
            qualification = "NEED_MORE_INFO"
        
        # Generate realistic user scenarios
        scenarios = self.scenario_templates.get(case_type, ["I need help with a family law matter."])
        
        for scenario in scenarios[:2]:  # Max 2 scenarios per case
            response = self._generate_chatbot_response(qualification, judgment_data)
            
            example = {
                "instruction": "You are a family law AI assistant. Determine if user needs case assessment, advisor consultation, or more information. Respond with appropriate guidance and qualification decision.",
                "input": scenario,
                "output": json.dumps({
                    "response": response,
                    "qualification": qualification,
                    "reasoning": f"Case involves {case_type.replace('_', ' ')} with {complexity} complexity",
                    "confidence": self._calculate_confidence(case_type, complexity),
                    "next_action": self._get_next_action(qualification)
                })
            }
            examples.append(example)
        
        return examples

    def _generate_chatbot_response(self, qualification: str, judgment_data: Dict[str, Any]) -> str:
        """Generate appropriate chatbot response based on case type"""
        case_type = judgment_data['case_type']
        main_issues = judgment_data['main_issues']
        
        case_descriptions = {
            'inheritance_family': 'inheritance and family provision matters',
            'child_arrangements': 'child custody and arrangements',
            'mental_capacity': 'mental capacity and Court of Protection matters',
            'international_family': 'international family law and cross-border disputes',
            'adoption_fostering': 'adoption and fostering procedures',
            'care_proceedings': 'care proceedings and child protection',
            'financial_remedy': 'financial remedy and property division',
            'divorce_dissolution': 'divorce and relationship dissolution',
            'domestic_violence': 'domestic violence and protection orders',
            'cohabitation': 'unmarried couples\' rights and cohabitation disputes'
        }
        
        case_description = case_descriptions.get(case_type, 'family law matters')
        
        if qualification == "QUALIFY_CASE":
            return f"This sounds like a complex case involving {case_description}. To give you the best guidance, I'd recommend completing our detailed assessment form so we can understand your full situation and provide accurate predictions about likely outcomes."
        
        elif qualification == "QUALIFY_ADVISOR":
            return f"Based on what you've mentioned about {case_description}, I'd recommend speaking directly with one of our qualified family law advisors who can provide personalized guidance for your specific situation."
        
        else:  # NEED_MORE_INFO
            return f"I'd like to help you better with your {case_description} query. Can you tell me more about your specific situation - for example, the key issues you're facing and what outcome you're hoping to achieve?"

    def _create_predictor_examples(self, judgment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create predictor training examples adapted for broader case types"""
        examples = []
        
        if not judgment_data['financial_info']['has_financial_data']:
            return examples
        
        # Create adapted Form E style input based on case type
        predictor_input = self._create_adapted_form_input(judgment_data)
        
        # Create prediction output based on decision
        prediction_output = self._create_prediction_output(judgment_data)
        
        example = {
            "instruction": f"Based on the {judgment_data['case_type'].replace('_', ' ')} case information provided, predict the likely court outcome and provide confidence assessment with legal reasoning.",
            "input": json.dumps(predictor_input),
            "output": json.dumps(prediction_output)
        }
        
        examples.append(example)
        return examples

    def _create_adapted_form_input(self, judgment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create adapted input format based on case type"""
        case_type = judgment_data['case_type']
        financial_info = judgment_data['financial_info']
        
        base_input = {
            "case_type": case_type.replace('_', ' ').title(),
            "main_issues": judgment_data['main_issues'],
            "complexity": judgment_data['complexity'],
            "financial_summary": financial_info['financial_summary']
        }
        
        # Add case-specific financial details
        case_specific = financial_info.get('case_specific_financial', {})
        for key, values in case_specific.items():
            if values:
                base_input[key] = values[:2]  # Include up to 2 examples
        
        return base_input

    def _create_prediction_output(self, judgment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create prediction output based on decision and case type"""
        decision_text = judgment_data['decision']
        case_type = judgment_data['case_type']
        
        # Extract outcome summary
        predicted_outcome = self._extract_outcome_summary(decision_text, case_type)
        
        # Calculate confidence
        confidence = self._calculate_prediction_confidence(judgment_data)
        
        # Extract key factors
        key_factors = self._extract_key_factors(judgment_data)
        
        prediction = {
            "predicted_outcome": predicted_outcome,
            "confidence": confidence,
            "key_factors": key_factors,
            "legal_reasoning": f"Based on {case_type.replace('_', ' ')} law and established precedents",
            "case_complexity": judgment_data['complexity']
        }
        
        # Add case-specific predictions
        if case_type == 'inheritance_family':
            prediction["inheritance_provision"] = "Reasonable provision likely to be awarded"
        elif case_type in ['financial_remedy', 'divorce_dissolution']:
            prediction["financial_arrangements"] = self._extract_financial_arrangements(decision_text)
        elif case_type == 'child_arrangements':
            prediction["child_arrangements"] = "Arrangements to be made in child's best interests"
        
        return prediction

    def _extract_outcome_summary(self, decision_text: str, case_type: str) -> str:
        """Extract outcome summary tailored to case type"""
        if not decision_text:
            outcomes = {
                'inheritance_family': 'Family provision claim to be determined based on dependency and estate size',
                'child_arrangements': 'Child arrangements to be determined in the best interests of the child',
                'mental_capacity': 'Best interests decision to be made considering all relevant factors',
                'international_family': 'Jurisdictional issues to be resolved according to international law',
                'financial_remedy': 'Financial arrangements to be determined based on needs and resources'
            }
            return outcomes.get(case_type, 'Court order addressing the legal issues presented')
        
        # Extract first few sentences as summary
        sentences = decision_text.split('. ')
        if len(sentences) > 3:
            summary = '. '.join(sentences[:3]) + '.'
        else:
            summary = decision_text
        
        # Limit length
        if len(summary) > 250:
            summary = summary[:250] + "..."
        
        return summary.strip()

    def _calculate_confidence(self, case_type: str, complexity: str) -> float:
        """Calculate confidence score for chatbot qualification"""
        base_confidence = 0.75
        
        # Adjust based on case type clarity
        high_confidence_types = ['financial_remedy', 'divorce_dissolution', 'child_arrangements']
        if case_type in high_confidence_types:
            base_confidence += 0.10
        
        # Adjust based on complexity
        if complexity == 'low':
            base_confidence += 0.10
        elif complexity == 'high':
            base_confidence += 0.05  # High complexity = more confident in needing case assessment
        
        return min(0.95, base_confidence)

    def _calculate_prediction_confidence(self, judgment_data: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on data quality"""
        base_confidence = 0.60
        
        # Increase confidence for clear financial data
        if judgment_data['financial_info']['has_financial_data']:
            base_confidence += 0.15
        
        # Increase confidence for substantial legal reasoning
        if len(judgment_data['legal_reasoning']) > 500:
            base_confidence += 0.10
        
        # Increase confidence for clear case type
        if judgment_data['case_type'] != 'unclassified':
            base_confidence += 0.10
        
        return min(0.90, base_confidence)

    def _extract_key_factors(self, judgment_data: Dict[str, Any]) -> List[str]:
        """Extract key factors based on case type"""
        case_type = judgment_data['case_type']
        factors = []
        
        # Case-specific factors
        factor_mapping = {
            'inheritance_family': ['Dependency on deceased', 'Estate size', 'Other beneficiaries', 'Reasonable provision'],
            'child_arrangements': ['Child welfare', 'Parental capacity', 'Housing stability', 'Relationship with child'],
            'mental_capacity': ['Capacity assessment', 'Best interests', 'Least restrictive option', 'Person\'s wishes'],
            'international_family': ['Jurisdictional issues', 'International treaties', 'Child\'s habitual residence', 'Enforcement'],
            'financial_remedy': ['Financial needs', 'Earning capacity', 'Standard of living', 'Duration of marriage'],
            'domestic_violence': ['Risk assessment', 'Safety concerns', 'Evidence of abuse', 'Protection measures']
        }
        
        factors = factor_mapping.get(case_type, ['Legal precedents', 'Case circumstances', 'Statutory criteria'])
        
        return factors[:4]  # Return max 4 factors

    def _extract_financial_arrangements(self, decision_text: str) -> Dict[str, str]:
        """Extract financial arrangements from decision"""
        arrangements = {}
        decision_lower = decision_text.lower()
        
        if "lump sum" in decision_lower:
            arrangements["lump_sum"] = "Lump sum payment ordered"
        if "periodical payment" in decision_lower or "maintenance" in decision_lower:
            arrangements["maintenance"] = "Periodical payments ordered"
        if "property" in decision_lower and ("transfer" in decision_lower or "sale" in decision_lower):
            arrangements["property"] = "Property adjustment ordered"
        if "pension" in decision_lower:
            arrangements["pension"] = "Pension provision addressed"
        
        return arrangements

    def _create_explainer_examples(self, judgment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create explainer training examples"""
        examples = []
        
        if len(judgment_data['legal_reasoning']) < 300:
            return examples
        
        explainer_input = {
            "case_summary": judgment_data['case_facts'][:500] + "..." if len(judgment_data['case_facts']) > 500 else judgment_data['case_facts'],
            "ai_prediction": self._extract_outcome_summary(judgment_data['decision'], judgment_data['case_type']),
            "main_issues": judgment_data['main_issues'],
            "case_type": judgment_data['case_type']
        }
        
        explainer_output = {
            "detailed_analysis": self._create_detailed_analysis(judgment_data),
            "legal_precedents": self._extract_precedents(judgment_data),
            "risk_factors": self._identify_risk_factors(judgment_data),
            "advisor_recommendations": self._generate_advisor_recommendations(judgment_data),
            "case_strengths": self._identify_case_strengths(judgment_data)
        }
        
        example = {
            "instruction": f"Provide detailed legal analysis and expert commentary for professional advisors reviewing this {judgment_data['case_type'].replace('_', ' ')} case. Include precedents, risk factors, and strategic recommendations.",
            "input": json.dumps(explainer_input),
            "output": json.dumps(explainer_output)
        }
        
        examples.append(example)
        return examples

    def _create_detailed_analysis(self, judgment_data: Dict[str, Any]) -> str:
        """Create detailed legal analysis"""
        reasoning = judgment_data['legal_reasoning']
        case_type = judgment_data['case_type']
        
        if len(reasoning) > 800:
            analysis = reasoning[:800] + f"... [Analysis continues with detailed examination of {case_type.replace('_', ' ')} law and its application to the case facts.]"
        else:
            analysis = reasoning + f" This analysis demonstrates the court's systematic approach to {case_type.replace('_', ' ')} law and balancing of relevant factors."
        
        return analysis

    def _extract_precedents(self, judgment_data: Dict[str, Any]) -> List[str]:
        """Extract legal precedents with case-type specific examples"""
        full_text = judgment_data['legal_reasoning']
        case_type = judgment_data['case_type']
        
        # Look for case citations
        case_patterns = [
            r'[A-Z][a-z]+ v\.? [A-Z][a-z]+',
            r'\[[0-9]{4}\] [A-Z]+ [0-9]+',
            r'\([0-9]{4}\) [A-Z]+ [0-9]+'
        ]
        
        precedents = []
        for pattern in case_patterns:
            matches = re.findall(pattern, full_text)
            precedents.extend(matches[:2])
        
        # Add typical precedents if none found
        if not precedents:
            precedent_examples = {
                'inheritance_family': ["Ilott v Mitson [2017] UKSC 73", "Cowan v Foreman [2019] EWCA Civ 1336"],
                'child_arrangements': ["Re B (Children) [2008] UKHL 35", "Re W (Children) [2012] EWCA Civ 999"],
                'mental_capacity': ["Re MN [2015] EWCA Civ 411", "Aintree v James [2013] UKSC 67"],
                'international_family': ["Re E (Children) [2011] UKSC 27", "Re L (Children) [2013] UKSC 75"],
                'financial_remedy': ["White v White [2001] 1 AC 596", "Miller v Miller [2006] UKHL 24"]
            }
            precedents = precedent_examples.get(case_type, ["Relevant case law applicable"])
        
        return precedents[:3]

    def _identify_risk_factors(self, judgment_data: Dict[str, Any]) -> List[str]:
        """Identify case-specific risk factors"""
        case_type = judgment_data['case_type']
        full_text = (judgment_data['case_facts'] + ' ' + judgment_data['legal_reasoning']).lower()
        
        risk_factors = {
            'inheritance_family': ['Will validity challenges', 'Family disputes over assets', 'Limitation period issues'],
            'child_arrangements': ['Parental alienation risks', 'International relocation', 'Child welfare concerns'],
            'mental_capacity': ['Capacity fluctuation', 'Family disagreements', 'Financial exploitation risks'],
            'international_family': ['Enforcement difficulties', 'Jurisdictional challenges', 'Treaty complications'],
            'financial_remedy': ['Asset disclosure issues', 'Pension complexities', 'Enforcement challenges'],
            'domestic_violence': ['Safety escalation', 'Breach of orders', 'Evidence gathering difficulties']
        }
        
        base_risks = risk_factors.get(case_type, ['Case complexity', 'Legal uncertainties'])
        
        # Add context-specific risks
        additional_risks = []
        if 'international' in full_text or 'foreign' in full_text:
            additional_risks.append('International enforcement complications')
        if 'business' in full_text or 'company' in full_text:
            additional_risks.append('Complex financial structures')
        if 'appeal' in full_text:
            additional_risks.append('Appeal proceedings risk')
        
        return (base_risks + additional_risks)[:4]

    def _generate_advisor_recommendations(self, judgment_data: Dict[str, Any]) -> List[str]:
        """Generate case-specific advisor recommendations"""
        case_type = judgment_data['case_type']
        complexity = judgment_data['complexity']
        
        recommendations = {
            'inheritance_family': [
                'Obtain detailed estate valuations',
                'Review dependency evidence thoroughly',
                'Consider alternative dispute resolution',
                'Assess limitation period carefully'
            ],
            'child_arrangements': [
                'Consider child welfare assessment',
                'Explore mediation options',
                'Document parenting capacity evidence',
                'Plan for future variation applications'
            ],
            'mental_capacity': [
                'Obtain current capacity assessment',
                'Consider least restrictive options',
                'Involve relevant family members',
                'Plan for best interests meetings'
            ],
            'international_family': [
                'Seek specialist international law advice',
                'Consider enforcement mechanisms',
                'Review treaty obligations',
                'Plan for jurisdictional challenges'
            ],
            'financial_remedy': [
                'Ensure full financial disclosure',
                'Obtain property valuations',
                'Consider pension sharing options',
                'Explore settlement negotiations'
            ]
        }
        
        base_recs = recommendations.get(case_type, ['Seek appropriate specialist advice'])
        
        # Add complexity-based recommendations
        if complexity == 'high':
            base_recs.append('Consider early case management hearing')
        
        return base_recs[:4]

    def _identify_case_strengths(self, judgment_data: Dict[str, Any]) -> List[str]:
        """Identify case strengths for strategic planning"""
        case_type = judgment_data['case_type']
        
        strengths = {
            'inheritance_family': ['Clear dependency evidence', 'Reasonable provision claim', 'Strong moral claim'],
            'child_arrangements': ['Child welfare focus', 'Stable home environment', 'Good parenting capacity'],
            'mental_capacity': ['Clear capacity assessment', 'Best interests evidence', 'Family support'],
            'international_family': ['Treaty protection', 'Clear jurisdiction', 'Enforcement mechanisms'],
            'financial_remedy': ['Full financial disclosure', 'Reasonable settlement position', 'Clear entitlement']
        }
        
        return strengths.get(case_type, ['Strong legal position'])[:3]

    def _get_next_action(self, qualification: str) -> str:
        """Get next action based on qualification"""
        action_map = {
            "QUALIFY_CASE": "form_submission",
            "QUALIFY_ADVISOR": "advisor_selection", 
            "NEED_MORE_INFO": "continue_conversation"
        }
        return action_map.get(qualification, "continue_conversation")

    def smart_sample_cases(self, xml_dir: Path, total_sample: int = 300) -> List[Path]:
        """Smart sampling based on dataset analysis to get optimal training mix"""
        xml_files = list(xml_dir.glob("*.xml"))
        
        if len(xml_files) <= total_sample:
            return xml_files
        
        # Sample strategy based on your dataset analysis
        target_distribution = {
            'inheritance_family': int(total_sample * 0.25),  # 75 cases - reduced from 34%
            'child_arrangements': int(total_sample * 0.25),  # 75 cases - increased from 20%
            'financial_remedy': int(total_sample * 0.15),    # 45 cases - increased from 4%
            'divorce_dissolution': int(total_sample * 0.10), # 30 cases - increased from 2%
            'mental_capacity': int(total_sample * 0.08),     # 24 cases - reduced from 12%
            'international_family': int(total_sample * 0.07), # 21 cases - reduced from 10%
            'adoption_fostering': int(total_sample * 0.05),  # 15 cases - reduced from 9%
            'domestic_violence': int(total_sample * 0.03),   # 9 cases - increased from 1%
            'care_proceedings': int(total_sample * 0.02),    # 6 cases - reduced from 4%
        }
        
        selected_files = []
        processed_count = 0
        
        # First pass: categorize files
        categorized_files = {case_type: [] for case_type in target_distribution.keys()}
        uncategorized = []
        
        print(f"Categorizing {len(xml_files)} files for smart sampling...")
        
        for xml_file in xml_files:
            try:
                # Quick text extraction for classification
                tree = ET.parse(xml_file)
                root = tree.getroot()
                text = self._extract_text_from_element(root).lower()
                
                case_type = self.classify_case_type(text)
                
                if case_type in categorized_files:
                    categorized_files[case_type].append(xml_file)
                else:
                    uncategorized.append(xml_file)
                
                processed_count += 1
                if processed_count % 500 == 0:
                    print(f"Categorized {processed_count}/{len(xml_files)} files...")
                    
            except Exception as e:
                uncategorized.append(xml_file)
                continue
        
        # Second pass: sample according to target distribution
        for case_type, target_count in target_distribution.items():
            available_files = categorized_files[case_type]
            
            if len(available_files) <= target_count:
                selected_files.extend(available_files)
                print(f"✅ {case_type}: selected {len(available_files)}/{target_count} (all available)")
            else:
                # Random sample from available files
                sampled = random.sample(available_files, target_count)
                selected_files.extend(sampled)
                print(f"✅ {case_type}: selected {target_count}/{len(available_files)}")
        
        # Fill remaining quota with uncategorized files
        remaining = total_sample - len(selected_files)
        if remaining > 0 and uncategorized:
            additional = random.sample(uncategorized, min(remaining, len(uncategorized)))
            selected_files.extend(additional)
            print(f"✅ Added {len(additional)} uncategorized files")
        
        print(f"\n🎯 Smart sampling complete: {len(selected_files)}/{total_sample} files selected")
        return selected_files

def main():
    """Main processing function with smart sampling"""
    processor = EnhancedLegalXMLProcessor()
    
    # Setup directories
    xml_dir = Path("data/raw/xml_judgments")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Smart sampling for optimal training data
    print("🎯 Starting smart case sampling...")
    selected_files = processor.smart_sample_cases(xml_dir, total_sample=300)
    
    all_training_data = {
        'chatbot': [],
        'predictor': [],
        'explainer': []
    }
    
    print(f"\n📊 Processing {len(selected_files)} selected files...")
    processed_count = 0
    
    for xml_file in selected_files:
        judgment_data = processor.extract_judgment_data(xml_file)
        
        if judgment_data:
            training_examples = processor.create_training_examples(judgment_data)
            
            # Categorize examples by component
            for example in training_examples:
                instruction = example['instruction']
                if 'assistant' in instruction and 'qualification' in instruction:
                    all_training_data['chatbot'].append(example)
                elif 'predict' in instruction.lower() or 'outcome' in instruction.lower():
                    all_training_data['predictor'].append(example)
                elif 'detailed' in instruction and 'advisor' in instruction:
                    all_training_data['explainer'].append(example)
            
            processed_count += 1
            
            if processed_count % 25 == 0:
                print(f"Processed {processed_count}/{len(selected_files)} files...")
    
    # Save training data for each component
    for component, examples in all_training_data.items():
        if examples:
            output_file = output_dir / f"{component}_training_data.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
            
            print(f"✅ Saved {len(examples)} {component} training examples to {output_file}")
    
    print(f"\n🎉 Smart processing complete!")
    print(f"📈 Processed {processed_count} files successfully")
    print(f"🧠 Total training examples generated: {sum(len(examples) for examples in all_training_data.values())}")

if __name__ == "__main__":
    main()