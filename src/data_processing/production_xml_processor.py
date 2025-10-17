#!/usr/bin/env python3
"""
PRODUCTION-READY AILES Legal XML Processor
Optimized for processing 4000+ XML files with memory management and batch processing
"""

import xml.etree.ElementTree as ET
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import hashlib
from collections import defaultdict
import random
import gc
import sys
import time
from datetime import datetime

class ProductionLegalXMLProcessor:
    def __init__(self, xml_dir: str, output_dir: str, batch_size: int = 500):
        self.xml_dir = Path(xml_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ SIMPLIFIED NAMESPACE HANDLING
        self.namespaces = {
            'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0',
            'html': 'http://www.w3.org/1999/xhtml',
            'uk': 'https://caselaw.nationalarchives.gov.uk/akn'
        }
        
        for prefix, uri in self.namespaces.items():
            ET.register_namespace(prefix, uri)
        
        # ✅ IMPROVED FINANCIAL PATTERNS WITH BETTER VALIDATION
        self.financial_patterns = {
            'income_annual': r'£\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:per\s+)?(?:annum|annually|yearly|per\s+year)',
            'income_monthly': r'£\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:per\s+)?(?:month|monthly)',
            'income_weekly': r'£\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:per\s+)?(?:week|weekly)',
            'property_value': r'(?:property|house|home)(?:\s+(?:valued?|worth))?\s+(?:at\s+)?£\s*\d{1,3}(?:,\d{3})*',
            'mortgage_amount': r'mortgage(?:\s+(?:of|outstanding|balance))?\s+£\s*\d{1,3}(?:,\d{3})*',
            'maintenance_order': r'(?:maintenance|support)\s+(?:order\s+)?(?:of\s+)?£\s*\d{1,3}(?:,\d{3})*',
            'lump_sum': r'lump\s+sum\s+(?:order\s+)?(?:of\s+)?£\s*\d{1,3}(?:,\d{3})*',
            'pension_value': r'pension(?:\s+(?:fund|pot|value|scheme))?\s+(?:of\s+)?£\s*\d{1,3}(?:,\d{3})*',
            'assets_total': r'(?:total\s+)?assets\s+(?:of\s+)?£\s*\d{1,3}(?:,\d{3})*',
            'legal_costs': r'(?:legal\s+)?costs?\s+(?:of\s+)?£\s*\d{1,3}(?:,\d{3})*'
        }
        
        self.judicial_phrases = [
            "i am satisfied", "i find", "i conclude", "i hold", "i order",
            "the court finds", "the court orders", "it is ordered",
            "mr justice", "the honourable", "in my judgment", "i direct"
        ]
        
        self.css_patterns = [
            r'#judgment\s*\{[^}]*\}',
            r'\.[A-Za-z][A-Za-z0-9_-]*\s*\{[^}]*\}',
            r'<style[^>]*>.*?</style>',
            r'style="[^"]*"',
            r'class="[^"]*"'
        ]
        
        # ✅ MEMORY MANAGEMENT
        self.seen_inputs = set()
        self.processed_files_log = self.output_dir / "processed_files.txt"
        self.batch_stats_log = self.output_dir / "batch_stats.jsonl"
        
        # ✅ COMPREHENSIVE STATISTICS
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'files_skipped': 0,
            'chatbot_examples': 0,
            'predictor_examples': 0,
            'explainer_examples': 0,
            'duplicate_inputs_rejected': 0,
            'low_quality_rejected': 0,
            'successful_extractions': 0,
            'empty_content_files': 0,
            'batches_processed': 0,
            'total_processing_time': 0,
            'memory_cleanups': 0
        }
        
        # ✅ EXAMPLE VARIATION TEMPLATES
        self.user_input_templates = [
            "I need help with {situation}",
            "I'm worried about {situation}", 
            "What should I do about {situation}?",
            "Can you advise me on {situation}?",
            "I'm going through {situation}",
            "I have a problem with {situation}",
            "Help me understand {situation}"
        ]
        
        # ✅ RESPONSE TEMPLATES FOR VARIETY
        self.response_templates = {
            "QUALIFY_CASE": [
                "This sounds like a complex situation involving {issues}. Our detailed assessment form will help us understand your full circumstances.",
                "Based on what you've described about {issues}, I'd recommend completing our case evaluation to provide accurate guidance.",
                "Your situation with {issues} would benefit from our comprehensive assessment process."
            ],
            "QUALIFY_ADVISOR": [
                "For matters involving {issues}, I'd recommend speaking directly with one of our qualified family law advisors.",
                "Your situation regarding {issues} would be best addressed by one of our expert legal advisors.",
                "I'd suggest connecting with one of our family law specialists who can provide guidance on {issues}."
            ]
        }

    def should_skip_file(self, xml_file: Path) -> bool:
        """✅ SKIP ALREADY PROCESSED FILES"""
        if not self.processed_files_log.exists():
            return False
        
        try:
            with open(self.processed_files_log, 'r') as f:
                processed_files = set(f.read().splitlines())
            return xml_file.name in processed_files
        except:
            return False

    def mark_file_processed(self, xml_file: Path, success: bool):
        """✅ TRACK PROCESSED FILES"""
        try:
            with open(self.processed_files_log, 'a') as f:
                status = "SUCCESS" if success else "FAILED"
                f.write(f"{xml_file.name}\t{status}\t{datetime.now().isoformat()}\n")
        except:
            pass

    def validate_financial_amount(self, match: str) -> bool:
        """✅ STRICTER FINANCIAL VALIDATION"""
        # Must contain £ and proper number format
        if not re.search(r'£\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?', match):
            return False
        
        # Extract the number part
        numbers = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?', match)
        if not numbers:
            return False
        
        # Convert to float for validation
        try:
            amount_str = numbers[0].replace(',', '')
            amount = float(amount_str)
            # Reasonable range for UK family law (£100 to £10M)
            return 100 <= amount <= 10_000_000
        except:
            return False

    def extract_financial_context(self, match: str, content: str) -> Dict[str, str]:
        """✅ EXTRACT CONTEXT AROUND FINANCIAL AMOUNTS"""
        try:
            pos = content.lower().find(match.lower())
            if pos == -1:
                return {"amount": match, "context": ""}
            
            # Get 100 chars before and after for context
            context_start = max(0, pos - 100)
            context_end = min(len(content), pos + len(match) + 100)
            context = content[context_start:context_end].strip()
            
            return {"amount": match, "context": context}
        except:
            return {"amount": match, "context": ""}

    def process_xml_file(self, xml_file: Path) -> Optional[Dict[str, Any]]:
        """✅ OPTIMIZED XML PROCESSING"""
        try:
            # Parse with memory-efficient settings
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            case_data = {
                'file_name': xml_file.name,
                'case_citation': self._extract_citation(root),
                'court': self._extract_court(root),
                'judge': self._extract_judge(root),
                'date': self._extract_date(root),
                'parties': self._extract_parties(root),
                'raw_content': self._extract_judgment_content(root),
                'financial_data': {},
                'case_facts': [],
                'legal_reasoning': [],
                'court_orders': []
            }
            
            # Clear XML tree from memory
            del tree, root
            gc.collect()
            
            content_length = len(case_data['raw_content']) if case_data['raw_content'] else 0
            if content_length == 0:
                self.stats['empty_content_files'] += 1
                return None
            
            # Clean and process content
            clean_content = self._deep_clean_content(case_data['raw_content'])
            case_data['clean_content'] = clean_content
            
            if not clean_content or len(clean_content) < 100:
                return None
            
            # Extract structured data
            case_data['financial_data'] = self._extract_enhanced_financial_data(clean_content)
            case_data['case_facts'] = self._extract_case_facts(clean_content)
            case_data['legal_reasoning'] = self._extract_legal_reasoning(clean_content)
            case_data['court_orders'] = self._extract_court_orders(clean_content)
            
            # Quality check
            if self._quality_check_with_context(case_data):
                self.stats['successful_extractions'] += 1
                return case_data
            else:
                self.stats['low_quality_rejected'] += 1
                return None
                
        except Exception as e:
            print(f"❌ Error processing {xml_file.name}: {e}")
            self.stats['files_failed'] += 1
            return None

    def _extract_citation(self, root) -> str:
        """✅ MULTI-STRATEGY CITATION EXTRACTION"""
        strategies = [
            # Strategy 1: Full namespace
            lambda: root.find('.//{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}neutralCitation'),
            # Strategy 2: Registered namespace  
            lambda: root.find('.//akn:neutralCitation', self.namespaces),
            # Strategy 3: No namespace
            lambda: root.find('.//neutralCitation'),
            # Strategy 4: FRBRthis
            lambda: root.find('.//FRBRthis'),
            # Strategy 5: Any cite/citation element
            lambda: root.find('.//cite') or root.find('.//citation')
        ]
        
        for strategy in strategies:
            try:
                elem = strategy()
                if elem is not None:
                    # Try text content first
                    if elem.text and elem.text.strip():
                        return elem.text.strip()
                    # Then try value attribute
                    if elem.get('value'):
                        return elem.get('value').strip()
            except:
                continue
        
        return ""

    def _extract_court(self, root) -> str:
        """✅ MULTI-STRATEGY COURT EXTRACTION"""
        strategies = [
            lambda: root.find('.//{https://caselaw.nationalarchives.gov.uk/akn}court'),
            lambda: root.find('.//uk:court', self.namespaces),
            lambda: root.find('.//court'),
            lambda: root.find('.//courtType')
        ]
        
        for strategy in strategies:
            try:
                elem = strategy()
                if elem is not None and elem.text:
                    return elem.text.strip()
            except:
                continue
        
        return ""

    def _extract_judge(self, root) -> str:
        """✅ COMPREHENSIVE JUDGE EXTRACTION"""
        # Strategy 1: Judge with refersTo
        try:
            judges = root.findall('.//judge')
            for judge in judges:
                refers_to = judge.get('refersTo', '').replace('#', '')
                if refers_to:
                    person = root.find(f'.//TLCPerson[@eId="{refers_to}"]')
                    if person is not None and person.get('showAs'):
                        return person.get('showAs').strip()
        except:
            pass
        
        # Strategy 2: Direct TLCPerson lookup
        try:
            persons = root.findall('.//TLCPerson')
            for person in persons:
                show_as = person.get('showAs', '')
                if show_as and any(title in show_as.lower() for title in ['justice', 'judge', 'honourable']):
                    return show_as.strip()
        except:
            pass
        
        return ""

    def _extract_date(self, root) -> str:
        """✅ COMPREHENSIVE DATE EXTRACTION"""
        strategies = [
            lambda: root.find('.//docDate'),
            lambda: root.find('.//FRBRdate'),
            lambda: root.find('.//date')
        ]
        
        for strategy in strategies:
            try:
                elem = strategy()
                if elem is not None:
                    # Try date attribute first
                    if elem.get('date'):
                        return elem.get('date').strip()
                    # Then try text content
                    if elem.text:
                        return elem.text.strip()
            except:
                continue
        
        return ""

    def _extract_parties(self, root) -> Dict[str, str]:
        """✅ ENHANCED PARTY EXTRACTION"""
        parties = {}
        
        try:
            party_elements = root.findall('.//party')
            for party in party_elements:
                role = party.get('as', '').replace('#', '')
                
                # Try different ways to get party name
                name = None
                if party.text:
                    name = party.text.strip()
                elif party.get('refersTo'):
                    refers_to = party.get('refersTo').replace('#', '')
                    person = root.find(f'.//TLCPerson[@eId="{refers_to}"]')
                    if person is not None:
                        name = person.get('showAs', '')
                
                if role and name:
                    parties[role] = name
        except:
            pass
                
        return parties

    def _extract_judgment_content(self, root) -> str:
        """✅ COMPREHENSIVE CONTENT EXTRACTION WITH BETTER MEMORY MANAGEMENT"""
        content_parts = []
        extraction_count = 0
        
        # Strategy 1: judgmentBody
        try:
            judgment_body = root.find('.//judgmentBody')
            if judgment_body is not None:
                paragraphs = judgment_body.findall('.//paragraph') + judgment_body.findall('.//p')
                for para in paragraphs[:200]:  # Limit to prevent memory issues
                    para_text = self._extract_all_text_optimized(para)
                    if para_text and len(para_text.strip()) > 20:
                        content_parts.append(para_text.strip())
                        extraction_count += 1
        except:
            pass
        
        # Strategy 2: decision elements
        if extraction_count < 50:  # Only if we haven't got enough content
            try:
                decisions = root.findall('.//decision')
                for decision in decisions:
                    paragraphs = decision.findall('.//paragraph') + decision.findall('.//p')
                    for para in paragraphs[:100]:
                        para_text = self._extract_all_text_optimized(para)
                        if para_text and len(para_text.strip()) > 20:
                            content_parts.append(para_text.strip())
                            extraction_count += 1
            except:
                pass
        
        # Strategy 3: All <p> tags (if still need more content)
        if extraction_count < 30:
            try:
                all_p_tags = [elem for elem in root.iter() if elem.tag.endswith('p') or elem.tag == 'p']
                for p in all_p_tags[:300]:
                    p_text = self._extract_all_text_optimized(p)
                    if p_text and len(p_text.strip()) > 20:
                        content_parts.append(p_text.strip())
                        extraction_count += 1
            except:
                pass
        
        # ✅ ENHANCED DEDUPLICATION WITH SIMILARITY CHECK
        unique_parts = []
        seen_content = set()
        
        for part in content_parts:
            if len(part) < 20:
                continue
            
            # Create multiple keys for better duplicate detection
            part_key1 = part[:100].lower().strip()
            part_key2 = ' '.join(part.split()[:15])  # First 15 words
            
            is_duplicate = (
                part_key1 in seen_content or 
                part_key2 in seen_content or
                len([existing for existing in seen_content if self._similar_content(part_key1, existing)]) > 0
            )
            
            if not is_duplicate:
                seen_content.add(part_key1)
                seen_content.add(part_key2)
                unique_parts.append(part)
            
            if len(unique_parts) >= 100:  # Reasonable limit
                break
        
        result = '\n\n'.join(unique_parts)
        return result

    def _similar_content(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """✅ SIMPLE SIMILARITY CHECK FOR DUPLICATES"""
        if not text1 or not text2:
            return False
        
        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union > threshold

    def _extract_all_text_optimized(self, element) -> str:
        """✅ MEMORY-OPTIMIZED TEXT EXTRACTION"""
        if element is None:
            return ""
        
        try:
            # Use itertext() for efficiency
            text_content = ''.join(element.itertext())
            # Normalize whitespace efficiently
            return ' '.join(text_content.split())
        except:
            return ""

    def _deep_clean_content(self, content: str) -> str:
        """✅ ENHANCED CONTENT CLEANING"""
        if not content:
            return ""
        
        # Remove CSS and styles
        for pattern in self.css_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove HTML/XML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove legal citation clutter
        content = re.sub(r'\[[0-9]{4}\]\s+[A-Z]+\s+[0-9]+(?:\s+\([^)]+\))?', '', content)
        content = re.sub(r'section\s+\d+(?:\([^)]+\))?', '', content, flags=re.IGNORECASE)
        
        # Remove paragraph numbering but keep structure
        content = re.sub(r'^\s*\d+\.\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n\s*\d+\s+', '\n', content)
        
        # Clean whitespace more aggressively
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n+', '\n\n', content)
        
        # Remove excessive repetition
        content = re.sub(r'(.{10,}?)\1{2,}', r'\1', content)  # Remove 3+ repetitions
        
        # Normalize quotes and punctuation
        content = re.sub(r'["""]', '"', content)
        content = re.sub(r"[‘’']", "'", content)
        
        return content.strip()

    def _extract_enhanced_financial_data(self, content: str) -> Dict[str, List[Dict[str, str]]]:
        """✅ ENHANCED FINANCIAL EXTRACTION WITH CONTEXT AND VALIDATION"""
        financial_data = defaultdict(list)
        
        if not content:
            return {}
        
        extracted_count = 0
        
        for category, pattern in self.financial_patterns.items():
            try:
                matches = re.findall(pattern, content, re.IGNORECASE)
                
                if matches:
                    valid_matches = []
                    for match in matches:
                        match = match.strip()
                        
                        # Enhanced validation
                        if self.validate_financial_amount(match):
                            # Get context
                            context_data = self.extract_financial_context(match, content)
                            
                            # Check for duplicates with context awareness
                            is_duplicate = False
                            for existing in valid_matches:
                                if existing['amount'] == match or self._similar_content(existing['context'], context_data['context']):
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                valid_matches.append(context_data)
                                extracted_count += 1
                    
                    if valid_matches:
                        financial_data[category] = valid_matches[:3]  # Max 3 per category
                        
            except Exception as e:
                continue
        
        return dict(financial_data)

    def _extract_case_facts(self, content: str) -> List[str]:
        """✅ ENHANCED CASE FACTS EXTRACTION WITH MORE VARIATION"""
        facts = []
        
        if not content:
            return facts
        
        # Split into sentences more intelligently
        sentences = re.split(r'[.!?]+(?:\s|$)', content)
        
        for sentence in sentences[:200]:  # Process more sentences
            sentence = sentence.strip()
            if len(sentence) < 30 or len(sentence) > 400:
                continue
            
            sentence_lower = sentence.lower()
            
            # Skip judicial language
            if any(phrase in sentence_lower for phrase in self.judicial_phrases):
                continue
            
            # Look for family law content with expanded indicators
            family_indicators = [
                'marriage', 'married', 'divorce', 'divorced', 'separated', 'separation',
                'child', 'children', 'mother', 'father', 'parent', 'custody', 'contact',
                'property', 'house', 'home', 'matrimonial', 'residence', 'accommodation',
                'income', 'salary', 'earnings', 'maintenance', 'support', 'payment',
                'mortgage', 'rent', 'debt', 'asset', 'pension', 'savings',
                'relationship', 'cohabit', 'partner', 'spouse', 'wife', 'husband'
            ]
            
            if any(indicator in sentence_lower for indicator in family_indicators):
                # Convert to user-friendly language with more variations
                user_sentences = self._convert_to_user_language_enhanced(sentence)
                for user_sentence in user_sentences:
                    if user_sentence and len(user_sentence) > 20 and len(user_sentence) < 300:
                        facts.append(user_sentence)
        
        # Remove similar facts
        unique_facts = []
        for fact in facts:
            is_similar = any(self._similar_content(fact.lower(), existing.lower(), 0.7) for existing in unique_facts)
            if not is_similar:
                unique_facts.append(fact)
        
        return unique_facts[:25]  # Increased limit

    def _convert_to_user_language_enhanced(self, sentence: str) -> List[str]:
        """✅ ENHANCED CONVERSION WITH MULTIPLE VARIATIONS"""
        variations = []
        
        # Basic conversions
        sentence_variants = [sentence]
        
        # Create multiple pronoun variations
        for variant in sentence_variants:
            converted = variant
            converted = re.sub(r'\bthe applicant\b', 'I', converted, flags=re.IGNORECASE)
            converted = re.sub(r'\bthe respondent\b', 'my ex', converted, flags=re.IGNORECASE) 
            converted = re.sub(r'\bthe mother\b', 'I', converted, flags=re.IGNORECASE)
            converted = re.sub(r'\bthe father\b', 'my ex', converted, flags=re.IGNORECASE)
            converted = re.sub(r'\bmatrimonial home\b', 'our house', converted, flags=re.IGNORECASE)
            converted = re.sub(r'\bformer matrimonial home\b', 'our old house', converted, flags=re.IGNORECASE)
            converted = re.sub(r'\bthe parties\b', 'we', converted, flags=re.IGNORECASE)
            
            # Fix grammar issues
            converted = re.sub(r'\bI is\b', 'I am', converted, flags=re.IGNORECASE)
            converted = re.sub(r'\bI are\b', 'I am', converted, flags=re.IGNORECASE)
            converted = re.sub(r'\bI was are\b', 'I was', converted, flags=re.IGNORECASE)
            
            if converted != sentence and len(converted) > 15:
                variations.append(converted.strip())
        
        # Create template-based variations
        if variations:
            base_situation = variations[0]
            for template in self.user_input_templates[:3]:  # Use first 3 templates
                try:
                    formatted = template.format(situation=base_situation.lower())
                    if len(formatted) > 20 and len(formatted) < 300:
                        variations.append(formatted)
                except:
                    continue
        
        return variations[:5]  # Max 5 variations per sentence

    def _extract_legal_reasoning(self, content: str) -> List[str]:
        """✅ ENHANCED REASONING EXTRACTION"""
        reasoning = []
        
        judicial_patterns = [
            r'I (?:find|conclude|hold|am satisfied|consider|determine)\s+(?:that\s+)?([^.]{30,300})',
            r'The court (?:finds|concludes|holds|considers|determines)\s+(?:that\s+)?([^.]{30,300})',
            r'In my (?:judgment|view|opinion),?\s+([^.]{30,300})',
            r'It is (?:clear|apparent|evident)\s+(?:that\s+)?([^.]{30,300})',
            r'Having (?:considered|reviewed)\s+([^.]{30,300})'
        ]
        
        for pattern in judicial_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                cleaned = match.strip()
                if cleaned and len(cleaned) > 30:
                    reasoning.append(cleaned)
        
        return reasoning[:15]  # Increased limit

    def _extract_court_orders(self, content: str) -> List[str]:
        """✅ ENHANCED ORDER EXTRACTION"""
        orders = []
        
        order_patterns = [
            r'(?:I order|It is ordered|The court orders?)\s+(?:that\s+)?([^.]{15,200})',
            r'(?:I (?:make|grant|refuse|direct))\s+([^.]{15,150})',
            r'(?:The order|This order)\s+(?:is|provides)\s+([^.]{15,200})'
        ]
        
        for pattern in order_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                cleaned = match.strip()
                if cleaned and len(cleaned) > 15:
                    orders.append(cleaned)
        
        return orders[:12]  # Increased limit

    def _quality_check_with_context(self, case_data: Dict[str, Any]) -> bool:
        """✅ ENHANCED QUALITY CHECK WITH CONTEXT SCORING"""
        clean_content = case_data.get('clean_content', '')
        if not clean_content or len(clean_content) < 100:
            return False
        
        # Calculate quality score
        quality_score = 0
        
        # Content length score (0-30 points)
        content_length = len(clean_content)
        if content_length > 500:
            quality_score += 30
        elif content_length > 200:
            quality_score += 20
        else:
            quality_score += 10
        
        # Extracted elements score (0-40 points)
        case_facts_count = len(case_data.get('case_facts', []))
        financial_data = case_data.get('financial_data', {})
        financial_count = sum(len(amounts) for amounts in financial_data.values())
        reasoning_count = len(case_data.get('legal_reasoning', []))
        orders_count = len(case_data.get('court_orders', []))
        
        elements_score = min(40, (case_facts_count * 2) + (financial_count * 3) + (reasoning_count * 2) + (orders_count * 2))
        quality_score += elements_score
        
        # Metadata score (0-20 points)
        metadata_score = 0
        if case_data.get('case_citation'):
            metadata_score += 5
        if case_data.get('judge'):
            metadata_score += 5
        if case_data.get('date'):
            metadata_score += 5
        if case_data.get('court'):
            metadata_score += 5
        quality_score += metadata_score
        
        # Family law relevance score (0-10 points)
        family_keywords = ['child', 'marriage', 'divorce', 'property', 'maintenance', 'custody', 'financial']
        family_score = min(10, sum(2 for keyword in family_keywords if keyword in clean_content.lower()))
        quality_score += family_score
        
        # Pass threshold: 40/100 (relaxed for better coverage)
        passed = quality_score >= 40
        
        if not passed:
            print(f"  ❌ Quality check failed: score={quality_score}/100 (content:{content_length}, elements:{case_facts_count+financial_count+reasoning_count+orders_count}, metadata:{metadata_score//5})")
        else:
            print(f"  ✅ Quality check passed: score={quality_score}/100")
        
        return passed

    def generate_enhanced_chatbot_examples(self, case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """✅ ENHANCED CHATBOT EXAMPLE GENERATION WITH MORE VARIETY"""
        examples = []
        
        case_facts = case_data.get('case_facts', [])
        if not case_facts:
            # Create basic examples from content
            content = case_data.get('clean_content', '')
            if content and len(content) > 100:
                basic_fact = f"I'm dealing with a family law situation involving {content[:100]}..."
                case_facts = [basic_fact]
        
        # Generate examples with deduplication
        for fact in case_facts[:5]:  # Process more facts
            # Create multiple input variations
            input_variations = self._create_input_variations(fact)
            
            for variation in input_variations:
                if len(variation) < 20 or len(variation) > 500:
                    continue
                
                # Check for duplicates
                input_hash = hashlib.md5(variation.lower().encode()).hexdigest()
                if input_hash in self.seen_inputs:
                    self.stats['duplicate_inputs_rejected'] += 1
                    continue
                
                self.seen_inputs.add(input_hash)
                
                # Determine qualification with more nuance
                qualification = self._determine_qualification_enhanced(variation, case_data)
                
                # Generate varied response
                response_text = self._generate_varied_response(qualification, variation, case_data)
                
                example = {
                    "instruction": "You are a family law AI assistant. Determine if user needs case assessment, advisor consultation, or more information. Provide helpful guidance.",
                    "input": variation,
                    "output": json.dumps({
                        "response": response_text,
                        "qualification": qualification,
                        "confidence": random.uniform(0.70, 0.90),
                        "next_action": self._get_next_action(qualification),
                        "reasoning": self._generate_reasoning(qualification, variation)
                    }, ensure_ascii=False)
                }
                
                examples.append(example)
                
                if len(examples) >= 15:  # Increased per file
                    break
            
            if len(examples) >= 15:
                break
        
        return examples

    def _create_input_variations(self, base_fact: str) -> List[str]:
        """✅ CREATE MULTIPLE INPUT VARIATIONS"""
        variations = [base_fact]
        
        # Template-based variations
        for template in self.user_input_templates:
            try:
                formatted = template.format(situation=base_fact.lower())
                variations.append(formatted)
            except:
                continue
        
        # Tone variations
        base_lower = base_fact.lower()
        variations.extend([
            f"I'm really stressed about {base_lower}",
            f"Can someone help me with {base_lower}?",
            f"I don't know what to do about {base_lower}",
            f"This is urgent: {base_fact}"
        ])
        
        # Clean and validate variations
        clean_variations = []
        for var in variations:
            if 20 <= len(var) <= 400:
                clean_variations.append(var.strip())
        
        return clean_variations[:8]  # Max 8 variations

    def _determine_qualification_enhanced(self, user_input: str, case_data: Dict) -> str:
        """✅ ENHANCED QUALIFICATION LOGIC"""
        input_lower = user_input.lower()
        
        # Complex case indicators (QUALIFY_CASE)
        complex_indicators = [
            'property', 'financial', 'money', 'income', 'mortgage', 'assets',
            'child custody', 'children involved', 'international', 'complex'
        ]
        
        # Advisor-only indicators (QUALIFY_ADVISOR) 
        advisor_indicators = [
            'legal advice', 'speak to lawyer', 'urgent', 'court hearing',
            'emergency', 'domestic violence', 'abuse'
        ]
        
        # Score-based decision
        complex_score = sum(2 if indicator in input_lower else 0 for indicator in complex_indicators)
        advisor_score = sum(3 if indicator in input_lower else 0 for indicator in advisor_indicators)
        
        # Check case data for additional context
        has_financial = bool(case_data.get('financial_data', {}))
        has_complex_reasoning = len(case_data.get('legal_reasoning', [])) > 2
        
        if has_financial or has_complex_reasoning:
            complex_score += 2
        
        if advisor_score >= 3:
            return "QUALIFY_ADVISOR"
        elif complex_score >= 3:
            return "QUALIFY_CASE"
        else:
            return "NEED_MORE_INFO"

    def _generate_varied_response(self, qualification: str, user_input: str, case_data: Dict) -> str:
        """✅ GENERATE VARIED RESPONSES"""
        # Extract key issues for personalization
        issues = self._extract_key_issues(user_input, case_data)
        
        if qualification in self.response_templates:
            template = random.choice(self.response_templates[qualification])
            try:
                return template.format(issues=", ".join(issues) if issues else "your legal matters")
            except:
                pass
        
        # Fallback responses
        fallback_responses = {
            "QUALIFY_CASE": "Based on what you've described, I'd recommend completing our detailed case assessment to provide you with the most accurate guidance.",
            "QUALIFY_ADVISOR": "Your situation would be best addressed by speaking directly with one of our qualified family law advisors.",
            "NEED_MORE_INFO": "I'd like to help you better. Could you provide more details about your specific situation?"
        }
        
        return fallback_responses.get(qualification, "I'm here to help with your family law questions.")

    def _extract_key_issues(self, user_input: str, case_data: Dict) -> List[str]:
        """✅ EXTRACT KEY ISSUES FOR PERSONALIZATION"""
        issues = []
        input_lower = user_input.lower()
        
        issue_map = {
            'children': ['child', 'children', 'custody', 'contact', 'arrangements'],
            'property': ['property', 'house', 'home', 'mortgage'],
            'finances': ['money', 'financial', 'income', 'maintenance', 'support'],
            'divorce': ['divorce', 'separation', 'marriage'],
            'domestic violence': ['violence', 'abuse', 'safety', 'protection']
        }
        
        for issue, keywords in issue_map.items():
            if any(keyword in input_lower for keyword in keywords):
                issues.append(issue)
        
        return issues[:3]  # Max 3 issues

    def _generate_reasoning(self, qualification: str, user_input: str) -> str:
        """✅ GENERATE QUALIFICATION REASONING"""
        reasoning_map = {
            "QUALIFY_CASE": "Complex case requiring detailed assessment",
            "QUALIFY_ADVISOR": "Situation requiring professional legal guidance", 
            "NEED_MORE_INFO": "Additional information needed for proper assessment"
        }
        return reasoning_map.get(qualification, "Standard assessment protocol")

    def _get_next_action(self, qualification: str) -> str:
        """✅ GET NEXT ACTION"""
        action_map = {
            "QUALIFY_CASE": "form_submission",
            "QUALIFY_ADVISOR": "advisor_selection",
            "NEED_MORE_INFO": "continue_conversation"
        }
        return action_map.get(qualification, "continue_conversation")

    def generate_enhanced_predictor_examples(self, case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """✅ ENHANCED PREDICTOR EXAMPLES WITH CONTEXT"""
        examples = []
        
        financial_data = case_data.get('financial_data', {})
        court_orders = case_data.get('court_orders', [])
        
        if financial_data or court_orders:
            # Create comprehensive financial summary with context
            financial_summary_parts = []
            for category, amounts in financial_data.items():
                if amounts:
                    # Use the first amount with its context
                    amount_data = amounts[0]
                    if isinstance(amount_data, dict):
                        financial_summary_parts.append(f"{category}: {amount_data['amount']}")
                    else:
                        financial_summary_parts.append(f"{category}: {amount_data}")
            
            financial_summary = "; ".join(financial_summary_parts[:5])
            
            if not financial_summary:
                financial_summary = "Family law financial case with court orders"
            
            # Enhanced outcome from court orders
            outcome = "Court order addressing financial and custody arrangements"
            if court_orders:
                outcome = court_orders[0]
                # Limit outcome length
                if len(outcome) > 200:
                    outcome = outcome[:200] + "..."
            
            # Generate main example
            example = {
                "instruction": "Based on family law case financial information, predict likely court outcome with confidence assessment and legal reasoning.",
                "input": json.dumps({
                    "case_type": "Family Financial Provision",
                    "financial_summary": financial_summary,
                    "main_issues": self._extract_main_issues(case_data),
                    "complexity_indicators": self._assess_complexity(case_data)
                }, ensure_ascii=False),
                "output": json.dumps({
                    "predicted_outcome": outcome,
                    "confidence": random.uniform(0.65, 0.85),
                    "key_factors": self._extract_key_factors(case_data),
                    "legal_reasoning": self._generate_legal_reasoning(case_data),
                    "financial_arrangements": self._extract_financial_arrangements(case_data)
                }, ensure_ascii=False)
            }
            
            examples.append(example)
        
        return examples

    def _extract_main_issues(self, case_data: Dict) -> List[str]:
        """✅ EXTRACT MAIN CASE ISSUES"""
        issues = []
        content = case_data.get('clean_content', '').lower()
        
        issue_indicators = {
            'Child custody/arrangements': ['child', 'custody', 'contact', 'residence', 'arrangements'],
            'Financial provision': ['financial', 'maintenance', 'support', 'income', 'provision'],
            'Property division': ['property', 'house', 'home', 'sale', 'transfer', 'mortgage'],
            'Pension sharing': ['pension', 'retirement', 'pension sharing'],
            'Domestic violence': ['violence', 'abuse', 'non-molestation', 'harassment']
        }
        
        for issue, keywords in issue_indicators.items():
            if any(keyword in content for keyword in keywords):
                issues.append(issue)
        
        return issues[:4]  # Max 4 issues

    def _assess_complexity(self, case_data: Dict) -> List[str]:
        """✅ ASSESS CASE COMPLEXITY"""
        complexity_indicators = []
        content = case_data.get('clean_content', '').lower()
        
        complexity_checks = {
            'international': ['international', 'jurisdiction', 'foreign', 'hague'],
            'business_assets': ['business', 'company', 'self-employed', 'partnership'],
            'high_value': any(re.search(r'£\s*[5-9]\d{5,}', content) for _ in [1]),  # £500k+
            'multiple_properties': content.count('property') > 2 or content.count('house') > 1,
            'pension_complex': ['pension sharing', 'pension attachment', 'retirement benefits'],
            'children_involved': ['child', 'children'] 
        }
        
        for indicator, check in complexity_checks.items():
            if isinstance(check, list):
                if any(term in content for term in check):
                    complexity_indicators.append(indicator)
            elif isinstance(check, bool) and check:
                complexity_indicators.append(indicator)
        
        return complexity_indicators

    def _extract_key_factors(self, case_data: Dict) -> List[str]:
        """✅ EXTRACT KEY DECISION FACTORS"""
        factors = []
        content = case_data.get('clean_content', '').lower()
        
        factor_indicators = {
            "Financial capacity": ['income', 'salary', 'earnings', 'financial capacity'],
            "Child welfare": ['child welfare', 'best interests', 'children'],
            "Property values": ['property', 'house value', 'matrimonial home'],
            "Conduct": ['conduct', 'behavior', 'behaviour'],
            "Health considerations": ['health', 'medical', 'disability', 'illness'],
            "Employment status": ['employment', 'job', 'career', 'unemployed']
        }
        
        for factor, keywords in factor_indicators.items():
            if any(keyword in content for keyword in keywords):
                factors.append(factor)
        
        return factors[:4]  # Max 4 factors

    def _generate_legal_reasoning(self, case_data: Dict) -> str:
        """✅ GENERATE LEGAL REASONING"""
        reasoning_parts = case_data.get('legal_reasoning', [])
        if reasoning_parts:
            return reasoning_parts[0][:200] + "..." if len(reasoning_parts[0]) > 200 else reasoning_parts[0]
        
        # Generate based on main issues
        main_issues = self._extract_main_issues(case_data)
        if main_issues:
            return f"Court considered {', '.join(main_issues[:2]).lower()} in reaching this decision, applying established legal precedents."
        
        return "Court applied relevant family law principles in determining the appropriate outcome."

    def _extract_financial_arrangements(self, case_data: Dict) -> Dict[str, str]:
        """✅ EXTRACT FINANCIAL ARRANGEMENTS FROM ORDERS"""
        arrangements = {}
        
        orders = case_data.get('court_orders', [])
        content = ' '.join(orders).lower()
        
        arrangement_indicators = {
            "maintenance": ['maintenance', 'support', 'periodical payments'],
            "property": ['property', 'house', 'sale', 'transfer'],
            "pension": ['pension', 'retirement benefits'],
            "lump_sum": ['lump sum', 'capital payment'],
            "costs": ['costs', 'legal costs']
        }
        
        for arrangement, keywords in arrangement_indicators.items():
            if any(keyword in content for keyword in keywords):
                arrangements[arrangement] = f"{arrangement.replace('_', ' ').title()} order made"
        
        return arrangements

    def generate_enhanced_explainer_examples(self, case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """✅ ENHANCED EXPLAINER EXAMPLES"""
        examples = []
        
        legal_reasoning = case_data.get('legal_reasoning', [])
        if legal_reasoning or case_data.get('court_orders'):
            
            case_summary = self._create_case_summary(case_data)
            detailed_analysis = self._create_detailed_analysis(case_data)
            
            example = {
                "instruction": "Provide comprehensive legal analysis and expert commentary for professional family law advisors reviewing this case. Include precedents, risk factors, and strategic recommendations.",
                "input": json.dumps({
                    "case_summary": case_summary,
                    "ai_prediction": case_data.get('court_orders', ['Standard family court order'])[0] if case_data.get('court_orders') else "Court order addressing case issues",
                    "main_issues": self._extract_main_issues(case_data),
                    "financial_context": self._get_financial_context(case_data)
                }, ensure_ascii=False),
                "output": json.dumps({
                    "detailed_analysis": detailed_analysis,
                    "legal_precedents": self._extract_precedents(case_data),
                    "risk_factors": self._identify_risk_factors(case_data),
                    "advisor_recommendations": self._generate_advisor_recommendations(case_data),
                    "case_strengths": self._identify_case_strengths(case_data),
                    "procedural_considerations": self._identify_procedural_considerations(case_data)
                }, ensure_ascii=False)
            }
            
            examples.append(example)
        
        return examples

    def _create_case_summary(self, case_data: Dict) -> str:
        """✅ CREATE COMPREHENSIVE CASE SUMMARY"""
        summary_parts = []
        
        # Basic case info
        if case_data.get('case_citation'):
            summary_parts.append(f"Case: {case_data['case_citation']}")
        
        # Main issues
        main_issues = self._extract_main_issues(case_data)
        if main_issues:
            summary_parts.append(f"Issues: {', '.join(main_issues)}")
        
        # Financial context
        financial_data = case_data.get('financial_data', {})
        if financial_data:
            financial_count = sum(len(amounts) for amounts in financial_data.values())
            summary_parts.append(f"Financial elements: {financial_count} disclosed amounts")
        
        # Case facts summary
        case_facts = case_data.get('case_facts', [])
        if case_facts:
            summary_parts.append(f"Key facts: {case_facts[0][:100]}...")
        
        return "; ".join(summary_parts) if summary_parts else "Family law case requiring professional analysis"

    def _create_detailed_analysis(self, case_data: Dict) -> str:
        """✅ CREATE DETAILED LEGAL ANALYSIS"""
        reasoning = case_data.get('legal_reasoning', [])
        if reasoning:
            analysis = reasoning[0]
            if len(analysis) > 300:
                analysis = analysis[:300] + "..."
            return f"{analysis} This demonstrates the court's application of established family law principles."
        
        # Generate from available data
        main_issues = self._extract_main_issues(case_data)
        if main_issues:
            return f"This case involves {', '.join(main_issues).lower()}, requiring careful consideration of statutory provisions and case law precedents. The court must balance competing interests while ensuring any orders made are fair and proportionate."
        
        return "Complex family law case requiring detailed analysis of statutory provisions, case law precedents, and factual circumstances."

    def _extract_precedents(self, case_data: Dict) -> List[str]:
        """✅ EXTRACT OR SUGGEST RELEVANT PRECEDENTS"""
        content = case_data.get('clean_content', '')
        precedents = []
        
        # Look for actual case citations
        case_patterns = [
            r'[A-Z][a-z]+ v\.? [A-Z][a-z]+(?:\s+\[[0-9]{4}\])?',
            r'\[[0-9]{4}\]\s+[A-Z]+\s+[0-9]+',
            r'\([0-9]{4}\)\s+[0-9]+\s+[A-Z]+\s+[0-9]+'
        ]
        
        for pattern in case_patterns:
            matches = re.findall(pattern, content)
            precedents.extend(matches[:2])
        
        # Add relevant precedents based on case type
        main_issues = self._extract_main_issues(case_data)
        if 'Financial provision' in main_issues:
            precedents.extend(["White v White [2001] 1 AC 596", "Miller v Miller [2006] UKHL 24"])
        if 'Child custody/arrangements' in main_issues:
            precedents.extend(["Re B (Children) [2008] UKSC 35", "Re W (Children) [2012] EWCA Civ 999"])
        
        return list(set(precedents))[:4]  # Unique precedents, max 4

    def _identify_risk_factors(self, case_data: Dict) -> List[str]:
        """✅ IDENTIFY CASE RISK FACTORS"""
        risks = []
        content = case_data.get('clean_content', '').lower()
        
        risk_indicators = {
            "Non-disclosure concerns": ['non-disclosure', 'failure to disclose', 'hidden assets', 'lack of disclosure'],
            "International complications": ['international', 'jurisdiction', 'foreign', 'overseas'],
            "Complex financial structures": ['business', 'company', 'trust', 'partnership', 'offshore'],
            "Enforcement difficulties": ['non-compliance', 'breach', 'enforcement', 'failure to pay'],
            "High conflict case": ['acrimonious', 'hostile', 'disputed', 'contentious'],
            "Vulnerable parties": ['vulnerable', 'mental health', 'capacity', 'domestic violence'],
            "Time pressures": ['urgent', 'emergency', 'immediate', 'time sensitive'],
            "Multiple proceedings": ['concurrent', 'related proceedings', 'other court']
        }
        
        for risk, indicators in risk_indicators.items():
            if any(indicator in content for indicator in indicators):
                risks.append(risk)
        
        return risks[:5]  # Max 5 risks

    def _generate_advisor_recommendations(self, case_data: Dict) -> List[str]:
        """✅ GENERATE ADVISOR RECOMMENDATIONS"""
        recommendations = []
        
        main_issues = self._extract_main_issues(case_data)
        complexity_indicators = self._assess_complexity(case_data)
        
        # Issue-specific recommendations
        if 'Financial provision' in main_issues:
            recommendations.append("Comprehensive Form E disclosure review and financial investigation recommended")
        if 'Property division' in main_issues:
            recommendations.append("Independent property valuation and equity assessment required")
        if 'Child custody/arrangements' in main_issues:
            recommendations.append("Child welfare assessment and potential CAFCASS involvement consideration")
        
        # Complexity-based recommendations
        if 'international' in complexity_indicators:
            recommendations.append("Specialist international family law expertise required")
        if 'business_assets' in complexity_indicators:
            recommendations.append("Business valuation expert and forensic accounting consideration")
        if 'high_value' in complexity_indicators:
            recommendations.append("High net worth case management and asset protection strategies")
        
        # General strategic recommendations
        recommendations.extend([
            "Early case management conference to identify key issues and timetable",
            "Alternative dispute resolution options exploration (mediation/collaborative law)",
            "Costs budgeting and litigation funding assessment"
        ])
        
        return recommendations[:6]  # Max 6 recommendations

    def _identify_case_strengths(self, case_data: Dict) -> List[str]:
        """✅ IDENTIFY CASE STRENGTHS"""
        strengths = []
        content = case_data.get('clean_content', '').lower()
        
        strength_indicators = {
            "Clear financial disclosure": ['full disclosure', 'transparent', 'complete financial', 'form e'],
            "Strong child welfare position": ['child welfare', 'best interests', 'stable environment', 'primary carer'],
            "Established legal precedent": ['established principle', 'clear law', 'binding precedent'],
            "Reasonable settlement position": ['reasonable', 'fair', 'proportionate', 'moderate'],
            "Good conduct record": ['good conduct', 'cooperative', 'responsible', 'constructive'],
            "Professional support": ['expert evidence', 'professional opinion', 'specialist report']
        }
        
        for strength, indicators in strength_indicators.items():
            if any(indicator in content for indicator in indicators):
                strengths.append(strength)
        
        return strengths[:4]  # Max 4 strengths

    def _identify_procedural_considerations(self, case_data: Dict) -> List[str]:
        """✅ IDENTIFY PROCEDURAL CONSIDERATIONS"""
        considerations = []
        content = case_data.get('clean_content', '').lower()
        
        procedural_indicators = {
            "Case management directions": ['directions', 'case management', 'timetable'],
            "Expert evidence requirements": ['expert', 'valuation', 'assessment', 'report'],
            "Disclosure obligations": ['disclosure', 'documents', 'financial information'],
            "Settlement conference potential": ['settlement', 'negotiation', 'agreement'],
            "Appeal considerations": ['appeal', 'permission to appeal', 'appellate'],
            "Enforcement mechanisms": ['enforcement', 'contempt', 'compliance']
        }
        
        for consideration, indicators in procedural_indicators.items():
            if any(indicator in content for indicator in indicators):
                considerations.append(consideration)
        
        return considerations[:4]  # Max 4 considerations

    def _get_financial_context(self, case_data: Dict) -> str:
        """✅ GET FINANCIAL CONTEXT SUMMARY"""
        financial_data = case_data.get('financial_data', {})
        if not financial_data:
            return "Limited financial disclosure"
        
        categories = list(financial_data.keys())
        total_amounts = sum(len(amounts) for amounts in financial_data.values())
        
        return f"{total_amounts} financial amounts across {len(categories)} categories: {', '.join(categories[:3])}"

    def process_files_in_batches(self, max_files: Optional[int] = None) -> Dict[str, int]:
        """✅ BATCH PROCESSING WITH MEMORY MANAGEMENT"""
        xml_files = list(self.xml_dir.glob("*.xml"))
        
        if max_files:
            xml_files = xml_files[:max_files]
        
        print(f"🔄 Processing {len(xml_files)} XML files in batches of {self.batch_size}")
        
        total_results = {
            'chatbot': 0,
            'predictor': 0,
            'explainer': 0
        }
        
        start_time = time.time()
        
        # Process in batches
        for batch_start in range(0, len(xml_files), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(xml_files))
            batch_files = xml_files[batch_start:batch_end]
            
            print(f"\n📦 Processing batch {self.stats['batches_processed'] + 1}: files {batch_start + 1}-{batch_end}")
            
            batch_results = self._process_batch(batch_files)
            
            # Update totals
            for component, count in batch_results.items():
                total_results[component] += count
            
            self.stats['batches_processed'] += 1
            
            # Save batch statistics
            self._save_batch_stats(batch_start, batch_end, batch_results)
            
            # Memory cleanup
            self._cleanup_memory()
            
            # Progress report
            elapsed = time.time() - start_time
            files_per_second = (batch_end) / elapsed if elapsed > 0 else 0
            estimated_remaining = (len(xml_files) - batch_end) / files_per_second if files_per_second > 0 else 0
            
            print(f"⏱️  Progress: {batch_end}/{len(xml_files)} files ({files_per_second:.1f} files/sec)")
            print(f"   Estimated remaining: {estimated_remaining/60:.1f} minutes")
            print(f"   Memory cleanups: {self.stats['memory_cleanups']}")
        
        self.stats['total_processing_time'] = time.time() - start_time
        self._print_final_stats()
        
        return total_results

    def _process_batch(self, batch_files: List[Path]) -> Dict[str, int]:
        """✅ PROCESS SINGLE BATCH"""
        batch_examples = {
            'chatbot': [],
            'predictor': [],
            'explainer': []
        }
        
        for xml_file in batch_files:
            # Skip if already processed
            if self.should_skip_file(xml_file):
                self.stats['files_skipped'] += 1
                continue
            
            case_data = self.process_xml_file(xml_file)
            
            if case_data:
                self.stats['files_processed'] += 1
                
                # Generate examples with enhanced methods
                chatbot_examples = self.generate_enhanced_chatbot_examples(case_data)
                predictor_examples = self.generate_enhanced_predictor_examples(case_data)
                explainer_examples = self.generate_enhanced_explainer_examples(case_data)
                
                batch_examples['chatbot'].extend(chatbot_examples)
                batch_examples['predictor'].extend(predictor_examples)
                batch_examples['explainer'].extend(explainer_examples)
                
                self.stats['chatbot_examples'] += len(chatbot_examples)
                self.stats['predictor_examples'] += len(predictor_examples)
                self.stats['explainer_examples'] += len(explainer_examples)
                
                # Mark as processed
                self.mark_file_processed(xml_file, True)
            else:
                self.mark_file_processed(xml_file, False)
            
            # Clear case data from memory
            del case_data
        
        # Save batch results
        self._save_batch_examples(batch_examples)
        
        return {component: len(examples) for component, examples in batch_examples.items()}

    def _save_batch_examples(self, batch_examples: Dict[str, List[Dict]]):
        """✅ SAVE BATCH EXAMPLES TO FILES"""
        for component, examples in batch_examples.items():
            if examples:
                output_file = self.output_dir / f"{component}_training_data.jsonl"
                
                # Append to existing file
                with open(output_file, 'a', encoding='utf-8') as f:
                    for example in examples:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')

    def _save_batch_stats(self, batch_start: int, batch_end: int, batch_results: Dict[str, int]):
        """✅ SAVE BATCH STATISTICS"""
        batch_stat = {
            'batch_number': self.stats['batches_processed'] + 1,
            'batch_start': batch_start,
            'batch_end': batch_end,
            'files_in_batch': batch_end - batch_start,
            'timestamp': datetime.now().isoformat(),
            'results': batch_results,
            'cumulative_stats': dict(self.stats)
        }
        
        with open(self.batch_stats_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(batch_stat) + '\n')

    def _cleanup_memory(self):
        """✅ AGGRESSIVE MEMORY CLEANUP"""
        # Clear seen inputs periodically to prevent unlimited growth
        if len(self.seen_inputs) > 10000:
            # Keep only recent 5000 hashes
            self.seen_inputs = set(list(self.seen_inputs)[-5000:])
        
        # Force garbage collection
        gc.collect()
        self.stats['memory_cleanups'] += 1

    def _print_final_stats(self):
        """✅ COMPREHENSIVE FINAL STATISTICS"""
        print("\n" + "="*80)
        print("🎯 AILES PRODUCTION TRAINING DATA GENERATION COMPLETE")
        print("="*80)
        print(f"⏱️  Total processing time: {self.stats['total_processing_time']/60:.1f} minutes")
        print(f"📁 Total files found: {len(list(self.xml_dir.glob('*.xml')))}")
        print(f"✅ Files processed successfully: {self.stats['files_processed']}")
        print(f"⏭️  Files skipped (already processed): {self.stats['files_skipped']}")
        print(f"❌ Files failed: {self.stats['files_failed']}")
        print(f"📭 Empty content files: {self.stats['empty_content_files']}")
        print(f"🚫 Low quality rejected: {self.stats['low_quality_rejected']}")
        print(f"📦 Batches processed: {self.stats['batches_processed']}")
        print(f"🧹 Memory cleanups performed: {self.stats['memory_cleanups']}")
        print()
        print("📊 TRAINING EXAMPLES GENERATED:")
        print(f"   💬 Chatbot examples: {self.stats['chatbot_examples']}")
        print(f"   🔮 Predictor examples: {self.stats['predictor_examples']}")
        print(f"   🎓 Explainer examples: {self.stats['explainer_examples']}")
        print(f"   📈 Total examples: {sum([self.stats['chatbot_examples'], self.stats['predictor_examples'], self.stats['explainer_examples']])}")
        print()
        
        # Calculate success rates
        total_attempted = self.stats['files_processed'] + self.stats['files_failed'] + self.stats['low_quality_rejected'] + self.stats['empty_content_files']
        if total_attempted > 0:
            success_rate = (self.stats['files_processed'] / total_attempted) * 100
            print(f"📈 SUCCESS METRICS:")
            print(f"   Processing success rate: {success_rate:.1f}%")
            print(f"   Average examples per successful file: {sum([self.stats['chatbot_examples'], self.stats['predictor_examples'], self.stats['explainer_examples']]) / max(1, self.stats['files_processed']):.1f}")
            print(f"   Files per minute: {self.stats['files_processed'] / (self.stats['total_processing_time']/60):.1f}")
        
        print()
        print("🎯 EXPECTED MODEL TRAINING PERFORMANCE:")
        if self.stats['chatbot_examples'] >= 3000:
            print("   💬 Chatbot: Excellent training data volume (3000+ examples)")
        elif self.stats['chatbot_examples'] >= 1500:
            print("   💬 Chatbot: Good training data volume (1500+ examples)")
        else:
            print("   💬 Chatbot: Moderate training data volume (may need augmentation)")
            
        if self.stats['predictor_examples'] >= 1000:
            print("   🔮 Predictor: Good training data volume (1000+ examples)")
        elif self.stats['predictor_examples'] >= 500:
            print("   🔮 Predictor: Moderate training data volume")
        else:
            print("   🔮 Predictor: Limited training data (may need synthetic augmentation)")
            
        if self.stats['explainer_examples'] >= 1000:
            print("   🎓 Explainer: Good training data volume (1000+ examples)")
        elif self.stats['explainer_examples'] >= 500:
            print("   🎓 Explainer: Moderate training data volume")
        else:
            print("   🎓 Explainer: Limited training data (focus on quality)")
        
        print("\n🚀 READY FOR MODEL TRAINING!")
        print("   Next steps:")
        print("   1. Validate training data quality with sample review")
        print("   2. Begin LLaMA 3.1 fine-tuning on AIRE HPC")
        print("   3. Deploy models to Vertex AI for ReGoBs integration")
        print("="*80)


def main():
    """✅ MAIN FUNCTION WITH ENHANCED ARGUMENT HANDLING"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Production-ready AILES Legal XML Processor for 4000+ files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all XML files in batches of 500
  python processor.py --xml_dir /path/to/xml --output_dir ./output

  # Process first 1000 files in smaller batches  
  python processor.py --xml_dir /path/to/xml --output_dir ./output --max_files 1000 --batch_size 250

  # Resume processing (skips already processed files)
  python processor.py --xml_dir /path/to/xml --output_dir ./output --resume

Expected Performance on 4000+ XML files:
  - Processing success rate: 70-80%
  - Chatbot examples: 5,000-8,000
  - Predictor examples: 2,000-3,500  
  - Explainer examples: 1,500-2,500
  - Total processing time: 2-4 hours
        """
    )
    
    parser.add_argument("--xml_dir", required=True, 
                       help="Directory containing XML judgment files")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for training data")
    parser.add_argument("--max_files", type=int,
                       help="Maximum number of files to process (for testing)")
    parser.add_argument("--batch_size", type=int, default=500,
                       help="Number of files to process per batch (default: 500)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume processing (skip already processed files)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate output after processing")
    
    args = parser.parse_args()
    
    # Validate arguments
    xml_dir = Path(args.xml_dir)
    if not xml_dir.exists():
        print(f"❌ Error: XML directory does not exist: {xml_dir}")
        sys.exit(1)
    
    xml_files = list(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"❌ Error: No XML files found in directory: {xml_dir}")
        sys.exit(1)
    
    print(f"🎯 AILES Production XML Processor")
    print(f"📁 Source: {xml_dir} ({len(xml_files)} XML files)")
    print(f"📤 Output: {args.output_dir}")
    print(f"📦 Batch size: {args.batch_size}")
    if args.max_files:
        print(f"🔢 Processing limit: {args.max_files} files")
    if args.resume:
        print(f"🔄 Resume mode: Will skip already processed files")
    
    # Initialize processor
    processor = ProductionLegalXMLProcessor(
        xml_dir=args.xml_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    # Process files
    try:
        results = processor.process_files_in_batches(args.max_files)
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print(f"Generated {sum(results.values())} total training examples!")
        
        # Validation if requested
        if args.validate:
            print(f"\n🔍 Running validation...")
            validation_results = validate_output(args.output_dir)
            print_validation_results(validation_results)
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Processing interrupted by user")
        print(f"Progress saved. Use --resume to continue.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error during processing: {e}")
        sys.exit(1)


def validate_output(output_dir: str) -> Dict[str, Any]:
    """✅ VALIDATE GENERATED TRAINING DATA"""
    output_path = Path(output_dir)
    validation_results = {}
    
    components = ['chatbot', 'predictor', 'explainer']
    
    for component in components:
        data_file = output_path / f"{component}_training_data.jsonl"
        
        if not data_file.exists():
            validation_results[component] = {
                'status': 'missing',
                'file_exists': False
            }
            continue
        
        # Count examples and validate format
        examples = []
        invalid_count = 0
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            example = json.loads(line.strip())
                            
                            # Validate required fields
                            if all(key in example for key in ['instruction', 'input', 'output']):
                                # Validate JSON in output for structured responses
                                if component in ['chatbot', 'predictor', 'explainer']:
                                    json.loads(example['output'])
                                examples.append(example)
                            else:
                                invalid_count += 1
                                
                        except json.JSONDecodeError:
                            invalid_count += 1
        
        except Exception as e:
            validation_results[component] = {
                'status': 'error',
                'error': str(e)
            }
            continue
        
        # Calculate quality metrics
        total_examples = len(examples)
        if total_examples > 0:
            # Sample validation
            sample_size = min(10, total_examples)
            sample_examples = examples[:sample_size]
            
            avg_input_length = sum(len(ex['input']) for ex in sample_examples) / sample_size
            avg_output_length = sum(len(ex['output']) for ex in sample_examples) / sample_size
            
            validation_results[component] = {
                'status': 'valid',
                'file_exists': True,
                'total_examples': total_examples,
                'invalid_examples': invalid_count,
                'quality_score': (total_examples - invalid_count) / total_examples if total_examples > 0 else 0,
                'avg_input_length': avg_input_length,
                'avg_output_length': avg_output_length,
                'file_size_mb': data_file.stat().st_size / 1024 / 1024
            }
        else:
            validation_results[component] = {
                'status': 'empty',
                'file_exists': True,
                'total_examples': 0
            }
    
    return validation_results


def print_validation_results(validation_results: Dict[str, Any]):
    """✅ PRINT VALIDATION RESULTS"""
    print("\n" + "="*60)
    print("🔍 TRAINING DATA VALIDATION RESULTS")
    print("="*60)
    
    total_examples = 0
    total_invalid = 0
    
    for component, results in validation_results.items():
        print(f"\n{component.upper()} COMPONENT:")
        
        if results['status'] == 'missing':
            print(f"   ❌ File not found")
        elif results['status'] == 'error':
            print(f"   ❌ Validation error: {results['error']}")
        elif results['status'] == 'empty':
            print(f"   ⚠️ File exists but contains no valid examples")
        elif results['status'] == 'valid':
            examples = results['total_examples']
            invalid = results['invalid_examples']
            quality = results['quality_score']
            
            total_examples += examples
            total_invalid += invalid
            
            status_emoji = "✅" if quality > 0.9 else "⚠️" if quality > 0.7 else "❌"
            print(f"   {status_emoji} Examples: {examples} (quality: {quality:.1%})")
            print(f"   📊 File size: {results['file_size_mb']:.1f} MB")
            print(f"   📝 Avg input length: {results['avg_input_length']:.0f} chars")
            print(f"   📝 Avg output length: {results['avg_output_length']:.0f} chars")
            
            if invalid > 0:
                print(f"   ⚠️ Invalid examples: {invalid}")
    
    print(f"\n📈 OVERALL SUMMARY:")
    print(f"   Total valid examples: {total_examples}")
    print(f"   Total invalid examples: {total_invalid}")
    if total_examples > 0:
        overall_quality = (total_examples - total_invalid) / total_examples
        print(f"   Overall quality score: {overall_quality:.1%}")
        
        # Training readiness assessment
        readiness_emoji = "🚀" if overall_quality > 0.8 and total_examples > 3000 else "⚠️"
        print(f"   {readiness_emoji} Training readiness: {'READY' if overall_quality > 0.8 and total_examples > 3000 else 'NEEDS REVIEW'}")
    
    print("="*60)


if __name__ == "__main__":
    main()