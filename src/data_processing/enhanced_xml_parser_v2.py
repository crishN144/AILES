#!/usr/bin/env python3
"""
AILES Legal AI - Production XML Parser v2.0
Addresses all critical issues and optimizes for best training data quality
"""

import xml.etree.ElementTree as ET
import json
import re
import argparse
import psutil
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import logging
import sys
import traceback
from contextlib import contextmanager
from collections import Counter, defaultdict
import threading
from queue import Queue
import yaml
import jsonschema
from jsonschema import validate
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('legal_parser.log')
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    # Load spaCy model for advanced NLP
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning(f"NLP dependencies not fully available: {e}")
    nlp = None

# JSON Schemas for validation
TRAINING_EXAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string", "minLength": 10},
        "input": {"type": "string", "minLength": 5},
        "output": {"type": "string", "minLength": 10},
        "component": {"type": "string", "enum": ["chatbot", "predictor", "explainer"]},
        "metadata": {
            "type": "object",
            "properties": {
                "case_type": {"type": "string"},
                "complexity_score": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "source_file": {"type": "string"},
                "extraction_quality": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
    },
    "required": ["instruction", "input", "output", "component", "metadata"]
}

@dataclass
class ProcessingConfig:
    """Configuration for XML processing"""
    max_file_size_mb: int = 100
    max_recursion_depth: int = 1000
    min_document_length: int = 100
    max_document_length: int = 50000
    min_paragraph_length: int = 20
    max_examples_per_file: int = 10
    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.8
    financial_amount_min: float = 1.0
    complexity_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.complexity_weights is None:
            self.complexity_weights = {
                'children': 0.3,
                'financial': 0.25,
                'property': 0.2,
                'international': 0.4,
                'business': 0.2,
                'violence': 0.35
            }

@dataclass
class QualityMetrics:
    """Track data quality metrics"""
    extraction_completeness: float = 0.0
    content_coherence: float = 0.0
    classification_confidence: float = 0.0
    financial_accuracy: float = 0.0
    structural_integrity: float = 0.0
    diversity_score: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate weighted overall quality score"""
        weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
        scores = [self.extraction_completeness, self.content_coherence, 
                 self.classification_confidence, self.financial_accuracy,
                 self.structural_integrity, self.diversity_score]
        return sum(w * s for w, s in zip(weights, scores))

class LegalJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for legal document data"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

class MemoryMonitor:
    """Monitor memory usage and prevent OOM"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        
    def check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.max_memory_percent:
            logger.warning(f"High memory usage: {memory_percent}%")
            return False
        return True
        
    @contextmanager
    def memory_guard(self):
        """Context manager for memory monitoring"""
        if not self.check_memory():
            raise MemoryError("Memory usage too high before operation")
        try:
            yield
        finally:
            if not self.check_memory():
                logger.warning("Memory usage high after operation")

class XMLStructureAnalyzer:
    """Analyze XML structure for optimal content extraction"""
    
    def __init__(self):
        self.structure_cache = {}
        
    def analyze_structure(self, root: ET.Element) -> Dict[str, Any]:
        """Analyze XML document structure"""
        file_hash = hashlib.md5(ET.tostring(root)).hexdigest()
        
        if file_hash in self.structure_cache:
            return self.structure_cache[file_hash]
            
        analysis = {
            'namespaces': self._extract_namespaces(root),
            'has_structured_content': self._has_structured_content(root),
            'judgment_structure': self._analyze_judgment_structure(root),
            'depth': self._calculate_depth(root, 0),
            'element_counts': self._count_elements(root)
        }
        
        self.structure_cache[file_hash] = analysis
        return analysis
        
    def _extract_namespaces(self, root: ET.Element) -> Dict[str, str]:
        """Dynamically extract namespaces"""
        namespaces = {}
        
        # Get namespaces from root element
        for prefix, uri in root.nsmap.items() if hasattr(root, 'nsmap') else {}:
            if prefix:  # Skip default namespace
                namespaces[prefix] = uri
                
        # Common legal document namespaces
        namespace_mapping = {
            'http://docs.oasis-open.org/legaldocml/ns/akn/3.0': 'akn',
            'http://www.w3.org/1999/xhtml': 'html',
            'https://caselaw.nationalarchives.gov.uk/akn': 'uk'
        }
        
        # Add standard namespaces
        for uri, prefix in namespace_mapping.items():
            if uri not in namespaces.values():
                namespaces[prefix] = uri
                
        return namespaces
        
    def _has_structured_content(self, root: ET.Element) -> bool:
        """Check if document has structured judgment content"""
        structured_indicators = [
            './/judgment', './/decision', './/judgmentBody',
            './/level', './/paragraph', './/content'
        ]
        
        for indicator in structured_indicators:
            if root.find(indicator) is not None:
                return True
        return False
        
    def _analyze_judgment_structure(self, root: ET.Element) -> Dict[str, List[str]]:
        """Analyze the structure of judgment sections"""
        structure = {
            'facts_sections': [],
            'reasoning_sections': [],
            'decision_sections': [],
            'procedural_sections': []
        }
        
        # Look for structured levels with headings
        levels = root.findall('.//level') + root.findall('.//section')
        
        for level in levels:
            level_id = level.get('eId', level.get('id', ''))
            heading_elem = level.find('.//heading') or level.find('.//h1') or level.find('.//h2')
            
            if heading_elem is not None and heading_elem.text:
                heading = heading_elem.text.lower().strip()
                
                # Classify sections by heading content
                if any(word in heading for word in ['fact', 'background', 'circumstances']):
                    structure['facts_sections'].append(level_id)
                elif any(word in heading for word in ['reasoning', 'analysis', 'discussion', 'law']):
                    structure['reasoning_sections'].append(level_id)
                elif any(word in heading for word in ['decision', 'order', 'conclusion', 'judgment']):
                    structure['decision_sections'].append(level_id)
                elif any(word in heading for word in ['procedure', 'hearing', 'application']):
                    structure['procedural_sections'].append(level_id)
                    
        return structure
        
    def _calculate_depth(self, element: ET.Element, current_depth: int) -> int:
        """Calculate maximum depth of XML tree"""
        if current_depth > 100:  # Prevent infinite recursion
            return current_depth
            
        max_child_depth = current_depth
        for child in element:
            child_depth = self._calculate_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
            
        return max_child_depth
        
    def _count_elements(self, root: ET.Element) -> Dict[str, int]:
        """Count different types of elements"""
        counts = defaultdict(int)
        
        def count_recursive(element):
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            counts[tag] += 1
            for child in element:
                count_recursive(child)
                
        count_recursive(root)
        return dict(counts)

class AdvancedContentExtractor:
    """Advanced content extraction with structure awareness"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.structure_analyzer = XMLStructureAnalyzer()
        self.stemmer = PorterStemmer() if 'nltk' in sys.modules else None
        self.stop_words = set(stopwords.words('english')) if 'nltk' in sys.modules else set()
        
    def extract_judgment_content(self, root: ET.Element) -> Dict[str, Any]:
        """Extract structured judgment content"""
        with MemoryMonitor().memory_guard():
            try:
                structure = self.structure_analyzer.analyze_structure(root)
                
                if structure['has_structured_content']:
                    return self._extract_structured_content(root, structure)
                else:
                    return self._extract_unstructured_content(root)
                    
            except ET.ParseError as e:
                logger.error(f"XML parsing error: {e}")
                raise
            except Exception as e:
                logger.error(f"Content extraction error: {e}")
                raise
                
    def _extract_structured_content(self, root: ET.Element, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from structured legal documents"""
        content = {
            'case_facts': '',
            'legal_reasoning': '',
            'decision': '',
            'procedural_history': '',
            'metadata': {},
            'quality_metrics': QualityMetrics()
        }
        
        namespaces = structure['namespaces']
        judgment_structure = structure['judgment_structure']
        
        # Extract facts
        facts_text = []
        for section_id in judgment_structure['facts_sections']:
            section = root.find(f".//*[@eId='{section_id}']") or root.find(f".//*[@id='{section_id}']")
            if section is not None:
                facts_text.append(self._extract_clean_text(section))
                
        # Extract reasoning
        reasoning_text = []
        for section_id in judgment_structure['reasoning_sections']:
            section = root.find(f".//*[@eId='{section_id}']") or root.find(f".//*[@id='{section_id}']")
            if section is not None:
                reasoning_text.append(self._extract_clean_text(section))
                
        # Extract decision
        decision_text = []
        for section_id in judgment_structure['decision_sections']:
            section = root.find(f".//*[@eId='{section_id}']") or root.find(f".//*[@id='{section_id}']")
            if section is not None:
                decision_text.append(self._extract_clean_text(section))
                
        # If structured extraction didn't work, fall back to heuristics
        if not any([facts_text, reasoning_text, decision_text]):
            return self._extract_by_heuristics(root)
            
        content.update({
            'case_facts': ' '.join(facts_text).strip(),
            'legal_reasoning': ' '.join(reasoning_text).strip(),
            'decision': ' '.join(decision_text).strip(),
            'procedural_history': self._extract_procedural_history(root, namespaces)
        })
        
        # Calculate quality metrics
        content['quality_metrics'] = self._calculate_content_quality(content)
        
        return content
        
    def _extract_unstructured_content(self, root: ET.Element) -> Dict[str, Any]:
        """Extract content from unstructured documents using advanced heuristics"""
        all_paragraphs = self._extract_paragraphs_with_context(root)
        
        if len(all_paragraphs) < 3:
            raise ValueError("Insufficient content for analysis")
            
        # Use NLP to classify paragraphs
        classified_content = self._classify_paragraphs_nlp(all_paragraphs)
        
        content = {
            'case_facts': classified_content['facts'],
            'legal_reasoning': classified_content['reasoning'],
            'decision': classified_content['decision'],
            'procedural_history': classified_content['procedural'],
            'quality_metrics': self._calculate_content_quality(classified_content)
        }
        
        return content
        
    def _classify_paragraphs_nlp(self, paragraphs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Use NLP to classify paragraph content"""
        if not nlp:
            return self._classify_paragraphs_heuristic(paragraphs)
            
        classified = {
            'facts': [],
            'reasoning': [],
            'decision': [],
            'procedural': []
        }
        
        # Define classification patterns
        patterns = {
            'facts': ['allegation', 'evidence', 'witness', 'occurred', 'happened', 'circumstances'],
            'reasoning': ['analysis', 'consider', 'conclude', 'principle', 'authority', 'precedent'],
            'decision': ['order', 'direct', 'find', 'declare', 'judgment', 'decree'],
            'procedural': ['hearing', 'application', 'motion', 'counsel', 'represent', 'court']
        }
        
        for para_info in paragraphs:
            text = para_info['text']
            doc = nlp(text) if len(text) < 1000000 else None  # Limit text length
            
            if doc is None:
                continue
                
            # Score each category
            scores = {}
            for category, keywords in patterns.items():
                score = 0
                for token in doc:
                    if token.lemma_.lower() in keywords:
                        score += 1
                    # Add entity-based scoring
                    if token.ent_type_ in ['PERSON', 'ORG', 'GPE'] and category == 'facts':
                        score += 0.5
                        
                scores[category] = score / len(doc) if len(doc) > 0 else 0
                
            # Assign to highest scoring category
            best_category = max(scores, key=scores.get) if scores else 'facts'
            classified[best_category].append(text)
            
        # Join classified content
        return {k: ' '.join(v) for k, v in classified.items()}
        
    def _classify_paragraphs_heuristic(self, paragraphs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Fallback heuristic classification"""
        total = len(paragraphs)
        
        # Improved heuristic based on paragraph position and content
        facts_paras = []
        reasoning_paras = []
        decision_paras = []
        
        for i, para_info in enumerate(paragraphs):
            text = para_info['text'].lower()
            position_ratio = i / total
            
            # Decision indicators (usually at end)
            if any(phrase in text for phrase in ['i order', 'i direct', 'i find', 'conclusion']):
                decision_paras.append(para_info['text'])
            # Early paragraphs likely facts
            elif position_ratio < 0.4:
                facts_paras.append(para_info['text'])
            # Middle paragraphs likely reasoning
            elif position_ratio < 0.8:
                reasoning_paras.append(para_info['text'])
            else:
                decision_paras.append(para_info['text'])
                
        return {
            'facts': ' '.join(facts_paras),
            'reasoning': ' '.join(reasoning_paras),
            'decision': ' '.join(decision_paras),
            'procedural': ''
        }
        
    def _extract_paragraphs_with_context(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract paragraphs with contextual information"""
        paragraphs = []
        
        # Find all paragraph-like elements
        para_elements = (root.findall('.//p') + root.findall('.//paragraph') + 
                        root.findall('.//content') + root.findall('.//text'))
        
        for i, elem in enumerate(para_elements):
            text = self._extract_clean_text(elem)
            
            if len(text.strip()) >= self.config.min_paragraph_length:
                paragraphs.append({
                    'text': text,
                    'position': i,
                    'element_type': elem.tag,
                    'attributes': dict(elem.attrib),
                    'parent_tag': elem.getparent().tag if elem.getparent() is not None else None
                })
                
        return paragraphs
        
    def _extract_clean_text(self, element: ET.Element, max_depth: int = None) -> str:
        """Safely extract text with recursion limits"""
        if max_depth is None:
            max_depth = self.config.max_recursion_depth
            
        def extract_recursive(elem, depth=0):
            if depth > max_depth:
                return ""
                
            text_parts = []
            
            if elem.text:
                text_parts.append(elem.text.strip())
                
            for child in elem:
                child_text = extract_recursive(child, depth + 1)
                if child_text:
                    text_parts.append(child_text)
                if child.tail:
                    text_parts.append(child.tail.strip())
                    
            return ' '.join(text_parts)
            
        return re.sub(r'\s+', ' ', extract_recursive(element)).strip()
        
    def _calculate_content_quality(self, content: Dict[str, Any]) -> QualityMetrics:
        """Calculate quality metrics for extracted content"""
        metrics = QualityMetrics()
        
        # Extraction completeness
        required_fields = ['case_facts', 'legal_reasoning', 'decision']
        filled_fields = sum(1 for field in required_fields if content.get(field, '').strip())
        metrics.extraction_completeness = filled_fields / len(required_fields)
        
        # Content coherence (using TF-IDF similarity)
        if all(content.get(field, '').strip() for field in required_fields):
            try:
                texts = [content[field] for field in required_fields]
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                # Average similarity indicates coherence
                metrics.content_coherence = np.mean(similarity_matrix[np.triu_indices(len(texts), k=1)])
            except Exception:
                metrics.content_coherence = 0.5  # Default moderate score
                
        # Structural integrity
        total_length = sum(len(content.get(field, '')) for field in required_fields)
        if total_length > 0:
            balance_score = 1.0 - np.std([len(content.get(field, '')) for field in required_fields]) / total_length
            metrics.structural_integrity = max(0, min(1, balance_score))
            
        return metrics

class SmartCaseClassifier:
    """Advanced case classification with confidence scoring"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.classification_cache = {}
        
        # Enhanced case type patterns with legal terminology
        self.case_patterns = {
            'inheritance_family': {
                'primary': ['inheritance act', 'family provision', 'reasonable provision', 'estate', 'will', 'testamentary'],
                'secondary': ['deceased', 'beneficiary', 'legacy', 'bequest', 'intestate'],
                'legal_refs': ['inheritance.*act', 'family.*provision.*act']
            },
            'child_arrangements': {
                'primary': ['child arrangements', 'contact', 'residence', 'custody', 'children act', 'parental responsibility'],
                'secondary': ['welfare', 'best interests', 'guardian', 'care order'],
                'legal_refs': ['children.*act.*1989', 'children.*act.*2004']
            },
            'mental_capacity': {
                'primary': ['mental capacity', 'court of protection', 'best interests', 'capacity act', 'mental health'],
                'secondary': ['lacking capacity', 'deputy', 'attorney', 'welfare'],
                'legal_refs': ['mental.*capacity.*act', 'mental.*health.*act']
            },
            'financial_remedy': {
                'primary': ['financial remedy', 'ancillary relief', 'matrimonial causes act', 'maintenance', 'periodical payments'],
                'secondary': ['lump sum', 'pension', 'property adjustment', 'spousal support'],
                'legal_refs': ['matrimonial.*causes.*act', 'divorce.*act']
            },
            'adoption_fostering': {
                'primary': ['adoption', 'placement order', 'special guardianship', 'foster', 'adoption act'],
                'secondary': ['birth parent', 'adoptive parent', 'freeing order', 'parental consent'],
                'legal_refs': ['adoption.*act.*1976', 'adoption.*act.*2002', 'children.*act']
            }
        }
        
    def classify_case(self, content: Dict[str, Any], legal_refs: List[str] = None) -> Tuple[str, float]:
        """Classify case type with confidence score"""
        text_content = ' '.join([
            content.get('case_facts', ''),
            content.get('legal_reasoning', ''),
            content.get('decision', '')
        ]).lower()
        
        # Check cache
        text_hash = hashlib.md5(text_content.encode()).hexdigest()
        if text_hash in self.classification_cache:
            return self.classification_cache[text_hash]
            
        scores = {}
        
        for case_type, patterns in self.case_patterns.items():
            score = 0
            
            # Primary pattern matching (higher weight)
            primary_matches = sum(1 for pattern in patterns['primary'] if pattern in text_content)
            score += primary_matches * 3
            
            # Secondary pattern matching
            secondary_matches = sum(1 for pattern in patterns['secondary'] if pattern in text_content)
            score += secondary_matches * 1
            
            # Legal reference matching (highest weight)
            if legal_refs:
                ref_text = ' '.join(legal_refs).lower()
                legal_matches = sum(1 for pattern in patterns['legal_refs'] if re.search(pattern, ref_text))
                score += legal_matches * 5
                
            # Normalize score
            total_patterns = len(patterns['primary']) * 3 + len(patterns['secondary']) + len(patterns['legal_refs']) * 5
            scores[case_type] = score / total_patterns if total_patterns > 0 else 0
            
        # Get best classification
        if not scores:
            result = ('unclassified', 0.0)
        else:
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            
            # Apply confidence threshold
            if confidence < self.config.confidence_threshold:
                result = ('unclassified', confidence)
            else:
                result = (best_type, confidence)
                
        self.classification_cache[text_hash] = result
        return result

class EnhancedFinancialExtractor:
    """Advanced financial information extraction"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
        # Comprehensive financial patterns with context
        self.financial_patterns = {
            'currency_amounts': {
                'pattern': r'£\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:million|thousand|k|m)?',
                'context': r'(?:income|salary|maintenance|lump sum|property|house|home|asset|pension|value|worth|cost|price|fee|award|order|pay|paid|receive|received)',
                'multipliers': {'million': 1000000, 'm': 1000000, 'thousand': 1000, 'k': 1000}
            },
            'property_values': {
                'pattern': r'(?:property|house|home|residence).*?valued?\s*(?:at)?\s*£\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                'context': r'(?:matrimonial|family|marital|joint|sole|beneficial)',
            },
            'maintenance_amounts': {
                'pattern': r'(?:maintenance|support|periodical\s+payments?|child\s+support).*?£\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per\s+(?:month|week|annum|year)|monthly|weekly|annually)?',
                'context': r'(?:order|ordered|direct|award|assess|payable|liable)'
            }
        }
        
    def extract_financial_data(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive financial information"""
        full_text = ' '.join([
            content.get('case_facts', ''),
            content.get('legal_reasoning', ''),
            content.get('decision', '')
        ])
        
        financial_data = {
            'has_financial_elements': False,
            'currency_amounts': [],
            'property_values': [],
            'maintenance_amounts': [],
            'total_amounts_mentioned': 0,
            'financial_complexity_score': 0.0
        }
        
        for category, pattern_info in self.financial_patterns.items():
            matches = self._extract_contextual_amounts(full_text, pattern_info)
            financial_data[category] = matches
            
            if matches:
                financial_data['has_financial_elements'] = True
                financial_data['total_amounts_mentioned'] += len(matches)
                
        # Calculate complexity score
        financial_data['financial_complexity_score'] = self._calculate_financial_complexity(financial_data)
        
        return financial_data
        
    def _extract_contextual_amounts(self, text: str, pattern_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract amounts with context validation"""
        amounts = []
        pattern = pattern_info['pattern']
        context_pattern = pattern_info.get('context', '')
        
        # Find all matches
        for match in re.finditer(pattern, text, re.IGNORECASE):
            amount_str = match.group(1)
            full_match = match.group(0)
            start, end = match.span()
            
            # Get surrounding context
            context_start = max(0, start - 100)
            context_end = min(len(text), end + 100)
            context = text[context_start:context_end]
            
            # Validate context if pattern provided
            if context_pattern and not re.search(context_pattern, context, re.IGNORECASE):
                continue
                
            # Parse amount
            try:
                amount = float(amount_str.replace(',', ''))
                
                # Apply multipliers
                multipliers = pattern_info.get('multipliers', {})
                for word, multiplier in multipliers.items():
                    if word in full_match.lower():
                        amount *= multiplier
                        break
                        
                if amount >= self.config.financial_amount_min:
                    amounts.append({
                        'amount': amount,
                        'formatted_amount': amount_str,
                        'context': context.strip(),
                        'confidence': self._calculate_amount_confidence(context, pattern_info)
                    })
                    
            except ValueError:
                continue
                
        return amounts
        
    def _calculate_amount_confidence(self, context: str, pattern_info: Dict[str, Any]) -> float:
        """Calculate confidence score for financial amounts"""
        base_confidence = 0.5
        
        # Increase confidence for strong context indicators
        strong_indicators = ['order', 'award', 'value', 'assess', 'direct']
        confidence_boost = sum(0.1 for indicator in strong_indicators if indicator in context.lower())
        
        return min(1.0, base_confidence + confidence_boost)
        
    def _calculate_financial_complexity(self, financial_data: Dict[str, Any]) -> float:
        """Calculate overall financial complexity score"""
        if not financial_data['has_financial_elements']:
            return 0.0
            
        complexity_factors = [
            min(1.0, financial_data['total_amounts_mentioned'] / 10),  # Number of amounts
            1.0 if financial_data['property_values'] else 0.0,  # Property involved
            1.0 if financial_data['maintenance_amounts'] else 0.0,  # Maintenance involved
        ]
        
        return sum(complexity_factors) / len(complexity_factors)

class IntelligentTrainingDataGenerator:
    """Generate high-quality, diverse training data"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.generated_examples = set()  # Track for diversity
        self.example_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, Any]:
        """Load diverse response templates"""
        return {
            'chatbot': {
                'QUALIFY_CASE': [
                    "Based on what you've described, this appears to be a complex {case_type} matter that would benefit from detailed legal assessment. I'd recommend completing our comprehensive case evaluation form.",
                    "This situation involves {case_type} issues that typically require specialized legal analysis. Let me guide you through our detailed assessment process.",
                    "From your description, this seems like a {case_type} case with multiple factors to consider. Our detailed evaluation would help provide you with accurate guidance."
                ],
                'QUALIFY_ADVISOR': [
                    "Your situation involves {case_type} elements that would benefit from professional guidance. I can connect you with a qualified family law advisor.",
                    "This appears to be a {case_type} matter where direct consultation with a legal professional would be valuable.",
                    "Based on the {case_type} aspects you've mentioned, I'd recommend speaking with one of our experienced advisors."
                ],
                'NEED_MORE_INFO': [
                    "I'd like to better understand your situation to provide the most helpful guidance. Could you tell me more about {specific_aspect}?",
                    "To give you the best advice, I need some additional information about {specific_aspect}.",
                    "Let me ask a few more questions about {specific_aspect} to better understand your circumstances."
                ]
            },
            'scenarios': {
                'inheritance_family': [
                    "My {relative} passed away {time_period} and I wasn't mentioned in the will, but I was financially dependent on them.",
                    "I believe my {relative}'s will doesn't make reasonable provision for me. I was {relationship_context} and expected to inherit.",
                    "My {relative} died without a will and I'm concerned about how their estate is being distributed among family members."
                ],
                'child_arrangements': [
                    "My ex-partner won't let me see our {age} child despite our previous arrangement.",
                    "We're separating and can't agree on where our children should live. What are my rights?",
                    "I'm concerned about my child's welfare when staying with their other parent."
                ],
                'financial_remedy': [
                    "We're divorcing and I'm worried about the financial settlement. My spouse earns significantly more than me.",
                    "I need advice about dividing our property and assets as part of our divorce proceedings.",
                    "My ex-spouse isn't paying the maintenance that was agreed. What can I do?"
                ]
            }
        }
        
    def generate_training_examples(self, content: Dict[str, Any], case_type: str, 
                                 classification_confidence: float, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate diverse, high-quality training examples"""
        examples = []
        
        if classification_confidence < self.config.confidence_threshold:
            logger.debug(f"Skipping example generation due to low confidence: {classification_confidence}")
            return examples
            
        # Generate chatbot examples
        chatbot_examples = self._generate_chatbot_examples(content, case_type, classification_confidence, metadata)
        examples.extend(chatbot_examples)
        
        # Generate predictor examples if sufficient financial data
        financial_data = metadata.get('financial_info', {})
        if financial_data.get('has_financial_elements', False):
            predictor_examples = self._generate_predictor_examples(content, case_type, financial_data, metadata)
            examples.extend(predictor_examples)
            
        # Generate explainer examples for complex cases
        complexity_score = metadata.get('complexity_score', 0)
        if complexity_score > 0.6:
            explainer_examples = self._generate_explainer_examples(content, case_type, complexity_score, metadata)
            examples.extend(explainer_examples)
            
        # Ensure diversity and quality
        examples = self._ensure_diversity(examples)
        examples = self._validate_examples(examples)
        
        return examples
        
    def _generate_chatbot_examples(self, content: Dict[str, Any], case_type: str, 
                                 confidence: float, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate diverse chatbot training examples"""
        examples = []
        
        # Determine qualification based on complexity and case type
        complexity_score = metadata.get('complexity_score', 0)
        qualification = self._determine_qualification(case_type, complexity_score)
        
        # Generate scenarios with variation
        base_scenarios = self.example_templates['scenarios'].get(case_type, [])
        if not base_scenarios:
            return examples
            
        # Create multiple examples with variation
        for i, scenario_template in enumerate(base_scenarios[:3]):  # Max 3 scenarios
            # Add variations to scenario
            varied_scenario = self._add_scenario_variations(scenario_template, case_type)
            
            # Generate response
            response_templates = self.example_templates['chatbot'][qualification]
            response_template = response_templates[i % len(response_templates)]
            response = response_template.format(
                case_type=case_type.replace('_', ' '),
                specific_aspect=self._get_specific_aspect(case_type)
            )
            
            # Create example
            example = {
                'instruction': "You are a family law AI assistant. Determine if user needs case assessment, advisor consultation, or more information.",
                'input': varied_scenario,
                'output': json.dumps({
                    'response': response,
                    'qualification': qualification,
                    'confidence': min(0.95, confidence + 0.1),
                    'next_action': self._get_next_action(qualification),
                    'case_type': case_type,
                    'complexity_indicators': metadata.get('complexity_indicators', [])
                }, cls=LegalJSONEncoder),
                'component': 'chatbot',
                'metadata': {
                    'case_type': case_type,
                    'complexity_score': complexity_score,
                    'confidence': confidence,
                    'source_file': metadata.get('file_name', ''),
                    'extraction_quality': metadata.get('quality_metrics', QualityMetrics()).overall_score()
                }
            }
            
            examples.append(example)
            
        return examples
        
    def _generate_predictor_examples(self, content: Dict[str, Any], case_type: str,
                                   financial_data: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate outcome prediction examples"""
        predictor_input = {
            'case_type': case_type.replace('_', ' ').title(),
            'main_issues': metadata.get('main_issues', []),
            'complexity_score': metadata.get('complexity_score', 0),
            'financial_complexity': financial_data.get('financial_complexity_score', 0),
            'has_property': bool(financial_data.get('property_values', [])),
            'has_maintenance': bool(financial_data.get('maintenance_amounts', []))
        }
        
        # Generate realistic outcome based on case type and complexity
        predicted_outcome = self._generate_realistic_outcome(case_type, predictor_input)
        
        example = {
            'instruction': "Based on the family law case information provided, predict the likely court outcome and key considerations.",
            'input': json.dumps(predictor_input, cls=LegalJSONEncoder),
            'output': json.dumps(predicted_outcome, cls=LegalJSONEncoder),
            'component': 'predictor',
            'metadata': {
                'case_type': case_type,
                'complexity_score': metadata.get('complexity_score', 0),
                'confidence': metadata.get('classification_confidence', 0.8),
                'source_file': metadata.get('file_name', ''),
                'extraction_quality': metadata.get('quality_metrics', QualityMetrics()).overall_score()
            }
        }
        
        return [example]
        
    def _generate_explainer_examples(self, content: Dict[str, Any], case_type: str,
                                   complexity_score: float, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed analysis for legal professionals"""
        
        # Extract key reasoning for analysis
        reasoning = content.get('legal_reasoning', '')
        if len(reasoning) > 1500:  # Truncate long content
            reasoning = reasoning[:1500] + "... [Analysis continues]"
            
        explainer_input = {
            'case_summary': content.get('case_facts', '')[:600],  # Limit length
            'main_issues': metadata.get('main_issues', []),
            'case_type': case_type,
            'complexity_indicators': metadata.get('complexity_indicators', [])
        }
        
        explainer_output = {
            'detailed_analysis': reasoning,
            'key_legal_principles': self._extract_legal_principles(reasoning),
            'risk_factors': self._identify_risk_factors(case_type, metadata),
            'advisor_recommendations': self._generate_advisor_recommendations(case_type, complexity_score),
            'precedent_relevance': "Analysis of relevant case law and statutory provisions",
            'complexity_assessment': {
                'overall_score': complexity_score,
                'key_factors': metadata.get('complexity_indicators', [])
            }
        }
        
        example = {
            'instruction': f"Provide detailed legal analysis for professional advisors reviewing this {case_type.replace('_', ' ')} case.",
            'input': json.dumps(explainer_input, cls=LegalJSONEncoder),
            'output': json.dumps(explainer_output, cls=LegalJSONEncoder),
            'component': 'explainer',
            'metadata': {
                'case_type': case_type,
                'complexity_score': complexity_score,
                'confidence': metadata.get('classification_confidence', 0.8),
                'source_file': metadata.get('file_name', ''),
                'extraction_quality': metadata.get('quality_metrics', QualityMetrics()).overall_score()
            }
        }
        
        return [example]
        
    def _ensure_diversity(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity in generated examples"""
        diverse_examples = []
        
        for example in examples:
            example_hash = hashlib.md5(
                (example['input'] + example['output']).encode()
            ).hexdigest()
            
            if example_hash not in self.generated_examples:
                self.generated_examples.add(example_hash)
                diverse_examples.append(example)
                
        return diverse_examples
        
    def _validate_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate examples against schema"""
        valid_examples = []
        
        for example in examples:
            try:
                validate(instance=example, schema=TRAINING_EXAMPLE_SCHEMA)
                valid_examples.append(example)
            except jsonschema.ValidationError as e:
                logger.warning(f"Example validation failed: {e}")
                continue
                
        return valid_examples
        
    # Helper methods for example generation
    def _determine_qualification(self, case_type: str, complexity_score: float) -> str:
        """Determine user qualification based on case complexity"""
        high_complexity_cases = ['inheritance_family', 'mental_capacity', 'international_family']
        
        if case_type in high_complexity_cases or complexity_score >= 0.7:
            return "QUALIFY_CASE"
        elif complexity_score >= 0.4:
            return "QUALIFY_ADVISOR"
        else:
            return "NEED_MORE_INFO"
            
    def _add_scenario_variations(self, scenario_template: str, case_type: str) -> str:
        """Add realistic variations to scenario templates"""
        variations = {
            'relative': ['father', 'mother', 'husband', 'wife', 'partner'],
            'time_period': ['last month', 'six months ago', 'recently', 'last year'],
            'relationship_context': ['their primary carer', 'financially dependent', 'living with them'],
            'age': ['5-year-old', '8-year-old', 'teenage', 'young']
        }
        
        import random
        for placeholder, options in variations.items():
            if f'{{{placeholder}}}' in scenario_template:
                scenario_template = scenario_template.replace(
                    f'{{{placeholder}}}', random.choice(options)
                )
                
        return scenario_template
        
    def _get_specific_aspect(self, case_type: str) -> str:
        """Get specific aspects to ask about for each case type"""
        aspects = {
            'inheritance_family': 'your relationship with the deceased and your financial circumstances',
            'child_arrangements': 'your current contact arrangements and any concerns about your child',
            'financial_remedy': 'your financial situation and assets to be considered',
            'mental_capacity': 'the specific decisions that need to be made',
            'adoption_fostering': 'the adoption process and your circumstances'
        }
        return aspects.get(case_type, 'your specific situation')
        
    def _get_next_action(self, qualification: str) -> str:
        """Get appropriate next action for qualification"""
        actions = {
            'QUALIFY_CASE': 'form_submission',
            'QUALIFY_ADVISOR': 'advisor_selection',
            'NEED_MORE_INFO': 'continue_conversation'
        }
        return actions.get(qualification, 'continue_conversation')
        
    def _generate_realistic_outcome(self, case_type: str, predictor_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic case outcome predictions"""
        outcomes = {
            'inheritance_family': "Court will assess reasonable provision claim under Inheritance Act, considering applicant's needs and estate resources",
            'financial_remedy': "Financial settlement considering statutory factors including income, needs, contributions and welfare of children",
            'child_arrangements': "Child arrangements order prioritizing child's welfare and maintaining relationship with both parents where safe",
            'mental_capacity': "Best interests decision required with input from relevant professionals and consideration of person's wishes"
        }
        
        base_outcome = outcomes.get(case_type, "Court order addressing the legal issues presented")
        
        return {
            'predicted_outcome': base_outcome,
            'confidence': 0.75 + (predictor_input.get('complexity_score', 0) * 0.2),
            'key_factors': predictor_input.get('main_issues', [])[:3],
            'legal_reasoning': f"Based on {case_type.replace('_', ' ')} law and established legal precedents",
            'risk_assessment': self._generate_risk_assessment(case_type, predictor_input),
            'timeline_estimate': self._estimate_timeline(case_type, predictor_input.get('complexity_score', 0))
        }
        
    def _extract_legal_principles(self, reasoning: str) -> List[str]:
        """Extract key legal principles from reasoning text"""
        # Simple pattern matching for legal principles
        principles = []
        
        principle_patterns = [
            r'the principle of ([^.]+)',
            r'established that ([^.]+)',
            r'the law requires ([^.]+)',
            r'the court must consider ([^.]+)'
        ]
        
        for pattern in principle_patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            principles.extend(matches[:2])  # Limit to avoid too many
            
        return principles[:5] if principles else ["Welfare of the child is paramount", "Statutory factors must be considered"]
        
    def _identify_risk_factors(self, case_type: str, metadata: Dict[str, Any]) -> List[str]:
        """Identify case-specific risk factors"""
        base_risks = {
            'inheritance_family': ["Complex family dynamics", "Estate valuation disputes", "Time limitations"],
            'child_arrangements': ["Conflict between parents", "Child's adjustment period", "Enforcement difficulties"],
            'financial_remedy': ["Asset disclosure issues", "Valuation complexities", "Enforcement challenges"],
            'mental_capacity': ["Fluctuating capacity", "Family disagreements", "Complex financial affairs"]
        }
        
        risks = base_risks.get(case_type, ["Complex legal issues", "Multiple parties involved"])
        
        # Add complexity-based risks
        complexity_indicators = metadata.get('complexity_indicators', [])
        if 'international' in complexity_indicators:
            risks.append("Cross-border enforcement issues")
        if 'violence' in complexity_indicators:
            risks.append("Safety and protection concerns")
            
        return risks
        
    def _generate_advisor_recommendations(self, case_type: str, complexity_score: float) -> List[str]:
        """Generate tailored advisor recommendations"""
        base_recommendations = [
            "Comprehensive case review and documentation",
            "Consider alternative dispute resolution options",
            "Ensure all relevant evidence is gathered"
        ]
        
        if complexity_score > 0.7:
            base_recommendations.extend([
                "Seek specialist counsel opinion",
                "Consider expert witness requirements"
            ])
            
        case_specific = {
            'inheritance_family': ["Review estate accounts and valuations", "Consider family mediation"],
            'child_arrangements': ["Child welfare assessment", "Consider CAFCASS involvement"],
            'financial_remedy': ["Complete financial disclosure", "Consider pension sharing implications"]
        }
        
        if case_type in case_specific:
            base_recommendations.extend(case_specific[case_type])
            
        return base_recommendations[:5]
        
    def _generate_risk_assessment(self, case_type: str, predictor_input: Dict[str, Any]) -> Dict[str, str]:
        """Generate risk assessment for case"""
        complexity = predictor_input.get('complexity_score', 0)
        
        if complexity >= 0.7:
            risk_level = "High"
            risk_desc = "Multiple complex factors requiring careful management"
        elif complexity >= 0.4:
            risk_level = "Medium"
            risk_desc = "Some challenging aspects requiring attention"
        else:
            risk_level = "Low"
            risk_desc = "Relatively straightforward matter"
            
        return {
            'overall_risk': risk_level,
            'description': risk_desc,
            'mitigation_required': "Yes" if complexity >= 0.5 else "Standard"
        }
        
    def _estimate_timeline(self, case_type: str, complexity_score: float) -> str:
        """Estimate case timeline"""
        base_timelines = {
            'inheritance_family': "6-18 months",
            'child_arrangements': "3-9 months",
            'financial_remedy': "9-18 months",
            'mental_capacity': "3-12 months",
            'adoption_fostering': "6-24 months"
        }
        
        base = base_timelines.get(case_type, "6-12 months")
        
        if complexity_score >= 0.7:
            return f"{base} (potentially longer due to complexity)"
        else:
            return base

class ProductionXMLProcessor:
    """Main processor class with all improvements"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.content_extractor = AdvancedContentExtractor(config)
        self.case_classifier = SmartCaseClassifier(config)
        self.financial_extractor = EnhancedFinancialExtractor(config)
        self.training_generator = IntelligentTrainingDataGenerator(config)
        
        # Statistics tracking
        self.processing_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'examples_generated': 0,
            'quality_scores': [],
            'case_type_distribution': Counter()
        }
        
    def validate_xml_file(self, xml_file: Path) -> bool:
        """Enhanced XML file validation"""
        try:
            # Basic file checks
            if not xml_file.exists() or xml_file.stat().st_size == 0:
                return False
                
            # Size check
            file_size_mb = xml_file.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                logger.warning(f"File too large: {xml_file} ({file_size_mb:.1f}MB)")
                return False
                
            # Parse test with error handling
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except ET.ParseError as e:
                logger.warning(f"XML parse error in {xml_file}: {e}")
                return False
            except Exception as e:
                logger.warning(f"Unexpected error parsing {xml_file}: {e}")
                return False
                
            # Structure validation
            if not self._validate_legal_document_structure(root):
                logger.debug(f"Invalid legal document structure: {xml_file}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Validation error for {xml_file}: {e}")
            return False
            
    def _validate_legal_document_structure(self, root: ET.Element) -> bool:
        """Validate that XML represents a legal document"""
        # Check for legal document indicators
        legal_indicators = [
            'akomantoso', 'judgment', 'decision', 'court', 'case',
            'legal', 'law', 'proceeding'
        ]
        
        root_tag = root.tag.lower()
        if any(indicator in root_tag for indicator in legal_indicators):
            return True
            
        # Check child elements
        for child in root:
            child_tag = child.tag.lower()
            if any(indicator in child_tag for indicator in legal_indicators):
                return True
                
        return False
        
    def extract_judgment_data(self, xml_file: Path) -> Optional[Dict[str, Any]]:
        """Enhanced judgment data extraction"""
        try:
            with self.memory_monitor.memory_guard():
                # Parse XML
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Initialize judgment data
                judgment_data = {
                    'file_name': xml_file.name,
                    'case_citation': '',
                    'court': '',
                    'judge': '',
                    'date': '',
                    'document_length': 0,
                    'extraction_timestamp': datetime.now().isoformat()
                }
                
                # Extract metadata
                self._extract_enhanced_metadata(root, judgment_data)
                
                # Extract content using advanced extractor
                content = self.content_extractor.extract_judgment_content(root)
                judgment_data.update(content)
                
                # Validate minimum content requirements
                total_content = len(judgment_data.get('case_facts', '')) + \
                              len(judgment_data.get('legal_reasoning', '')) + \
                              len(judgment_data.get('decision', ''))
                              
                if total_content < self.config.min_document_length:
                    logger.debug(f"Insufficient content in {xml_file}: {total_content} chars")
                    return None
                    
                judgment_data['document_length'] = total_content
                
                # Extract legal references
                legal_refs = self._extract_legal_references(root)
                
                # Classify case type
                case_type, classification_confidence = self.case_classifier.classify_case(
                    judgment_data, legal_refs
                )
                judgment_data['case_type'] = case_type
                judgment_data['classification_confidence'] = classification_confidence
                
                # Extract financial information
                financial_data = self.financial_extractor.extract_financial_data(judgment_data)
                judgment_data['financial_info'] = financial_data
                judgment_data['has_financial_elements'] = financial_data['has_financial_elements']
                
                # Calculate complexity
                complexity_data = self._calculate_advanced_complexity(judgment_data, financial_data)
                judgment_data.update(complexity_data)
                
                # Update statistics
                self.processing_stats['files_processed'] += 1
                self.processing_stats['case_type_distribution'][case_type] += 1
                if hasattr(judgment_data.get('quality_metrics'), 'overall_score'):
                    self.processing_stats['quality_scores'].append(
                        judgment_data['quality_metrics'].overall_score()
                    )
                
                return judgment_data
                
        except Exception as e:
            logger.error(f"Error extracting data from {xml_file}: {e}")
            logger.debug(traceback.format_exc())
            self.processing_stats['files_failed'] += 1
            return None
            
    def _extract_enhanced_metadata(self, root: ET.Element, judgment_data: Dict[str, Any]):
        """Extract comprehensive metadata"""
        try:
            # Dynamic namespace detection
            structure = self.content_extractor.structure_analyzer.analyze_structure(root)
            namespaces = structure['namespaces']
            
            # Extract citation with multiple fallback patterns
            citation_patterns = [
                f'.//{ns}:FRBRthis' for ns in namespaces.keys()
            ] + [
                './/*[@value]',
                './/citation',
                './/*[contains(@class,"citation")]',
                './/*[contains(text(),"[")]'
            ]
            
            for pattern in citation_patterns:
                try:
                    elem = root.find(pattern, namespaces if ':' in pattern else None)
                    if elem is not None:
                        citation = elem.get('value') or elem.text
                        if citation and '[' in citation:
                            judgment_data['case_citation'] = citation.strip()
                            break
                except:
                    continue
                    
            # Extract judge information
            judge_patterns = [
                f'.//{ns}:TLCPerson[@showAs]' for ns in namespaces.keys()
            ] + [
                './/judge',
                './/*[contains(@class,"judge")]',
                './/*[contains(text(),"JUSTICE")]'
            ]
            
            for pattern in judge_patterns:
                try:
                    elem = root.find(pattern, namespaces if ':' in pattern else None)
                    if elem is not None:
                        judge = elem.get('showAs') or elem.text
                        if judge:
                            judgment_data['judge'] = judge.strip()
                            break
                except:
                    continue
                    
            # Extract court information
            court_patterns = [
                f'.//{ns}:court' for ns in namespaces.keys()
            ] + [
                './/*[contains(@class,"court")]',
                './/*[contains(text(),"COURT")]'
            ]
            
            for pattern in court_patterns:
                try:
                    elem = root.find(pattern, namespaces if ':' in pattern else None)
                    if elem is not None:
                        court = elem.text or elem.get('value')
                        if court:
                            judgment_data['court'] = court.strip()
                            break
                except:
                    continue
                    
        except Exception as e:
            logger.debug(f"Metadata extraction warning: {e}")
            
    def _extract_legal_references(self, root: ET.Element) -> List[str]:
        """Extract legal references for improved classification"""
        legal_refs = []
        
        try:
            # Look for legal reference elements
            ref_patterns = [
                './/ref[@type="legislation"]',
                './/ref[@uk:type="legislation"]',
                './/*[contains(@class,"legislation")]',
                './/*[contains(text(),"Act")]'
            ]
            
            for pattern in ref_patterns:
                refs = root.findall(pattern)
                for ref in refs:
                    ref_text = ref.get('canonical') or ref.get('uk:canonical') or ref.text
                    if ref_text:
                        legal_refs.append(ref_text.strip())
                        
        except Exception as e:
            logger.debug(f"Legal reference extraction warning: {e}")
            
        return legal_refs
        
    def _calculate_advanced_complexity(self, judgment_data: Dict[str, Any], 
                                     financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive complexity metrics"""
        complexity_indicators = []
        complexity_scores = {}
        
        # Content-based complexity
        content_length = judgment_data.get('document_length', 0)
        content_complexity = min(1.0, content_length / 10000)  # Normalize to max 10k chars
        complexity_scores['content'] = content_complexity
        
        # Financial complexity
        financial_complexity = financial_data.get('financial_complexity_score', 0)
        complexity_scores['financial'] = financial_complexity
        if financial_complexity > 0.3:
            complexity_indicators.append('financial')
            
        # Case type complexity
        case_type = judgment_data.get('case_type', '')
        high_complexity_types = ['inheritance_family', 'mental_capacity', 'international_family']
        case_complexity = 0.8 if case_type in high_complexity_types else 0.4
        complexity_scores['case_type'] = case_complexity
        
        # Content analysis complexity
        content_text = ' '.join([
            judgment_data.get('case_facts', ''),
            judgment_data.get('legal_reasoning', ''),
            judgment_data.get('decision', '')
        ]).lower()
        
        # Check for complexity indicators
        complexity_keywords = {
            'children': ['child', 'children', 'minor', 'custody', 'welfare'],
            'property': ['property', 'house', 'home', 'real estate', 'matrimonial home'],
            'international': ['international', 'foreign', 'hague', 'jurisdiction'],
            'business': ['business', 'company', 'partnership', 'commercial'],
            'violence': ['violence', 'abuse', 'harassment', 'protection', 'molestation'],
            'procedural': ['appeal', 'cross-appeal', 'interlocutory', 'summary judgment']
        }
        
        for category, keywords in complexity_keywords.items():
            if any(keyword in content_text for keyword in keywords):
                complexity_indicators.append(category)
                weight = self.config.complexity_weights.get(category, 0.2)
                complexity_scores[category] = weight
                
        # Calculate overall complexity
        overall_complexity = sum(complexity_scores.values()) / len(complexity_scores) if complexity_scores else 0
        overall_complexity = min(1.0, overall_complexity)  # Cap at 1.0
        
        # Determine main issues from indicators
        main_issues = []
        issue_mapping = {
            'children': 'Child arrangements and welfare',
            'financial': 'Financial provision and assets',
            'property': 'Property division and ownership',
            'international': 'International jurisdiction and enforcement',
            'business': 'Business interests and commercial assets',
            'violence': 'Domestic violence and protection',
            'procedural': 'Procedural and appeal matters'
        }
        
        for indicator in complexity_indicators:
            if indicator in issue_mapping:
                main_issues.append(issue_mapping[indicator])
                
        return {
            'complexity_indicators': complexity_indicators,
            'complexity_score': overall_complexity,
            'complexity_breakdown': complexity_scores,
            'main_issues': main_issues
        }
        
    def create_training_examples(self, judgment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create high-quality training examples"""
        try:
            # Validate input data quality
            if not self._validate_judgment_data_quality(judgment_data):
                logger.debug(f"Skipping example generation for {judgment_data.get('file_name', 'unknown')} due to quality issues")
                return []
                
            # Extract metadata for example generation
            metadata = {
                'file_name': judgment_data.get('file_name', ''),
                'case_type': judgment_data.get('case_type', ''),
                'classification_confidence': judgment_data.get('classification_confidence', 0),
                'complexity_score': judgment_data.get('complexity_score', 0),
                'complexity_indicators': judgment_data.get('complexity_indicators', []),
                'main_issues': judgment_data.get('main_issues', []),
                'financial_info': judgment_data.get('financial_info', {}),
                'quality_metrics': judgment_data.get('quality_metrics', QualityMetrics())
            }
            
            # Generate examples
            examples = self.training_generator.generate_training_examples(
                judgment_data,
                metadata['case_type'],
                metadata['classification_confidence'],
                metadata
            )
            
            self.processing_stats['examples_generated'] += len(examples)
            return examples
            
        except Exception as e:
            logger.error(f"Error creating training examples: {e}")
            return []
            
    def _validate_judgment_data_quality(self, judgment_data: Dict[str, Any]) -> bool:
        """Validate that judgment data meets quality standards"""
        required_fields = ['case_facts', 'legal_reasoning', 'decision']
        
        # Check required fields
        for field in required_fields:
            if not judgment_data.get(field, '').strip():
                return False
                
        # Check minimum content length
        total_content = sum(len(judgment_data.get(field, '')) for field in required_fields)
        if total_content < self.config.min_document_length:
            return False
            
        # Check classification confidence
        classification_confidence = judgment_data.get('classification_confidence', 0)
        if classification_confidence < self.config.confidence_threshold:
            return False
            
        return True
        
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = self.processing_stats.copy()
        
        # Calculate additional metrics
        total_files = stats['files_processed'] + stats['files_failed']
        stats['success_rate'] = stats['files_processed'] / total_files if total_files > 0 else 0
        
        if stats['quality_scores']:
            stats['average_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
            stats['quality_distribution'] = {
                'high': sum(1 for s in stats['quality_scores'] if s >= 0.8) / len(stats['quality_scores']),
                'medium': sum(1 for s in stats['quality_scores'] if 0.5 <= s < 0.8) / len(stats['quality_scores']),
                'low': sum(1 for s in stats['quality_scores'] if s < 0.5) / len(stats['quality_scores'])
            }
        
        # Convert Counter to dict for JSON serialization
        stats['case_type_distribution'] = dict(stats['case_type_distribution'])
        
        return stats

class ThreadSafeFileWriter:
    """Thread-safe file writing with locking"""
    
    def __init__(self):
        self._locks = defaultdict(threading.Lock)
        
    def write_examples_to_file(self, examples: List[Dict[str, Any]], 
                              batch_id: int, output_dir: Path) -> Dict[str, int]:
        """Thread-safe writing of examples to batch files"""
        component_counts = defaultdict(int)
        
        # Group examples by component
        examples_by_component = defaultdict(list)
        for example in examples:
            component = example.pop('component', 'unknown')
            examples_by_component[component].append(example)
            
        # Write each component's examples
        for component, component_examples in examples_by_component.items():
            batch_file = output_dir / f"{component}_batch_{batch_id:04d}.jsonl"
            
            with self._locks[str(batch_file)]:
                try:
                    with open(batch_file, 'a', encoding='utf-8', buffering=1) as f:
                        for example in component_examples:
                            # Validate example before writing
                            try:
                                validate(instance=example, schema=TRAINING_EXAMPLE_SCHEMA)
                                f.write(json.dumps(example, cls=LegalJSONEncoder, ensure_ascii=False) + '\n')
                                component_counts[component] += 1
                            except jsonschema.ValidationError as e:
                                logger.warning(f"Skipping invalid example: {e}")
                                continue
                            except Exception as e:
                                logger.error(f"Error writing example: {e}")
                                continue
                                
                except IOError as e:
                    logger.error(f"Failed to write to {batch_file}: {e}")
                    continue
                    
        return dict(component_counts)

def process_file_batch(batch_info: Tuple[List[Path], int, ProcessingConfig]) -> Tuple[Dict, int, List[Dict]]:
    """Process a batch of XML files with enhanced error handling"""
    file_paths, batch_id, config = batch_info
    
    # Initialize processor for this batch
    processor = ProductionXMLProcessor(config)
    
    stats = {
        'total_files': len(file_paths),
        'processed_files': 0,
        'failed_files': 0,
        'skipped_files': 0,
        'examples_generated': 0,
        'processing_errors': []
    }
    
    all_examples = []
    
    for xml_file in file_paths:
        try:
            # Memory check before processing each file
            if not processor.memory_monitor.check_memory():
                logger.warning(f"Memory usage too high, skipping remaining files in batch {batch_id}")
                stats['skipped_files'] += len(file_paths) - len(all_examples)
                break
                
            # Validate file
            if not processor.validate_xml_file(xml_file):
                stats['failed_files'] += 1
                stats['processing_errors'].append({
                    'file': str(xml_file),
                    'error': 'Validation failed',
                    'type': 'validation_error'
                })
                continue
                
            # Extract judgment data
            start_time = time.time()
            judgment_data = processor.extract_judgment_data(xml_file)
            processing_time = time.time() - start_time
            
            if judgment_data is None:
                stats['failed_files'] += 1
                stats['processing_errors'].append({
                    'file': str(xml_file),
                    'error': 'Data extraction failed',
                    'type': 'extraction_error',
                    'processing_time': processing_time
                })
                continue
                
            # Create training examples
            examples = processor.create_training_examples(judgment_data)
            
            if examples:
                all_examples.extend(examples)
                stats['processed_files'] += 1
                stats['examples_generated'] += len(examples)
                
                # Log successful processing
                logger.debug(f"Processed {xml_file}: {len(examples)} examples, "
                           f"{judgment_data.get('case_type', 'unknown')} case, "
                           f"{processing_time:.2f}s")
            else:
                stats['failed_files'] += 1
                stats['processing_errors'].append({
                    'file': str(xml_file),
                    'error': 'No examples generated',
                    'type': 'example_generation_error',
                    'processing_time': processing_time
                })
                
        except MemoryError as e:
            logger.error(f"Memory error processing {xml_file}: {e}")
            stats['failed_files'] += 1
            stats['processing_errors'].append({
                'file': str(xml_file),
                'error': str(e),
                'type': 'memory_error'
            })
            break  # Stop processing this batch
            
        except Exception as e:
            logger.error(f"Unexpected error processing {xml_file}: {e}")
            logger.debug(traceback.format_exc())
            stats['failed_files'] += 1
            stats['processing_errors'].append({
                'file': str(xml_file),
                'error': str(e),
                'type': 'unexpected_error'
            })
            continue
            
    logger.info(f"Batch {batch_id} completed: {stats['processed_files']}/{stats['total_files']} files, "
               f"{stats['examples_generated']} examples generated")
               
    return stats, batch_id, all_examples

def merge_batch_files(output_dir: Path, components: List[str], config: ProcessingConfig):
    """Enhanced batch file merging with validation and deduplication"""
    logger.info("Merging batch files...")
    
    for component in components:
        batch_files = sorted(output_dir.glob(f"{component}_batch_*.jsonl"))
        if not batch_files:
            logger.info(f"No batch files found for {component}")
            continue
            
        final_file = output_dir / f"{component}_training_data.jsonl"
        seen_examples = set()  # For deduplication
        valid_examples = 0
        invalid_examples = 0
        duplicate_examples = 0
        
        try:
            with open(final_file, 'w', encoding='utf-8', buffering=8192) as outf:
                for batch_file in batch_files:
                    try:
                        with open(batch_file, 'r', encoding='utf-8') as inf:
                            for line_num, line in enumerate(inf, 1):
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                try:
                                    # Parse and validate example
                                    example = json.loads(line)
                                    validate(instance=example, schema=TRAINING_EXAMPLE_SCHEMA)
                                    
                                    # Check for duplicates
                                    example_hash = hashlib.md5(
                                        (example['input'] + example['output']).encode()
                                    ).hexdigest()
                                    
                                    if example_hash in seen_examples:
                                        duplicate_examples += 1
                                        continue
                                        
                                    seen_examples.add(example_hash)
                                    outf.write(line + '\n')
                                    valid_examples += 1
                                    
                                except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                                    logger.warning(f"Invalid example in {batch_file}:{line_num}: {e}")
                                    invalid_examples += 1
                                    continue
                                    
                        # Clean up batch file
                        batch_file.unlink()
                        
                    except Exception as e:
                        logger.error(f"Error processing batch file {batch_file}: {e}")
                        continue
                        
            logger.info(f"Created {final_file}: {valid_examples} valid examples, "
                       f"{invalid_examples} invalid, {duplicate_examples} duplicates removed")
                       
        except Exception as e:
            logger.error(f"Error creating final file {final_file}: {e}")

def save_processing_report(stats: Dict[str, Any], output_dir: Path, config: ProcessingConfig):
    """Save comprehensive processing report"""
    report = {
        'processing_summary': stats,
        'configuration': asdict(config),
        'timestamp': datetime.now().isoformat(),
        'recommendations': generate_recommendations(stats)
    }
    
    report_file = output_dir / 'processing_report.json'
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, cls=LegalJSONEncoder)
        logger.info(f"Processing report saved to {report_file}")
    except Exception as e:
        logger.error(f"Failed to save processing report: {e}")

def generate_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on processing statistics"""
    recommendations = []
    
    success_rate = stats.get('success_rate', 0)
    if success_rate < 0.8:
        recommendations.append(f"Low success rate ({success_rate:.1%}). Review validation criteria and input data quality.")
        
    avg_quality = stats.get('average_quality_score', 0)
    if avg_quality < 0.7:
        recommendations.append(f"Average quality score is low ({avg_quality:.2f}). Consider improving content extraction.")
        
    case_distribution = stats.get('case_type_distribution', {})
    if len(case_distribution) < 3:
        recommendations.append("Limited case type diversity. Consider expanding input dataset.")
        
    if 'unclassified' in case_distribution and case_distribution['unclassified'] > 0.2:
        recommendations.append("High unclassified case rate. Review classification patterns and thresholds.")
        
    examples_per_file = stats.get('examples_generated', 0) / max(1, stats.get('files_processed', 1))
    if examples_per_file < 2:
        recommendations.append(f"Low examples per file ({examples_per_file:.1f}). Review example generation criteria.")
        
    return recommendations

def main():
    """Enhanced main function with comprehensive configuration and monitoring"""
    parser = argparse.ArgumentParser(
        description="Production XML processor for AILES Legal AI v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument("--input_dir", default="data/raw", 
                       help="Input directory containing XML files")
    parser.add_argument("--output_dir", default="data/processed", 
                       help="Output directory for processed data")
    parser.add_argument("--config_file", 
                       help="YAML configuration file path")
    
    # Processing arguments
    parser.add_argument("--max_files", type=int, 
                       help="Maximum files to process (for testing)")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=50, 
                       help="Files per batch")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from previous progress")
    
    # Quality control arguments
    parser.add_argument("--min_quality", type=float, default=0.5,
                       help="Minimum quality score for examples")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                       help="Minimum classification confidence")
    parser.add_argument("--max_memory_percent", type=float, default=80.0,
                       help="Maximum memory usage percentage")
    
    # Advanced options
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate files without processing")
    parser.add_argument("--sample_rate", type=float, default=1.0,
                       help="Random sample rate for testing (0.0-1.0)")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    if args.config_file and Path(args.config_file).exists():
        try:
            with open(args.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            config = ProcessingConfig(**config_data)
            logger.info(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
            config = ProcessingConfig()
    else:
        config = ProcessingConfig()
        
    # Override config with command line arguments
    if args.confidence_threshold:
        config.confidence_threshold = args.confidence_threshold
    if args.max_memory_percent:
        # Create memory monitor with custom threshold
        pass
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    checkpoint_file = output_dir / "progress.json"
    file_writer = ThreadSafeFileWriter()
    
    logger.info(f"Starting AILES Legal AI XML Processor v2.0")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Workers: {args.workers}, Batch size: {args.batch_size}")
    
    # Load progress if resuming
    processed_files = set()
    if args.resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                processed_files = set(json.load(f))
            logger.info(f"Resuming: loaded {len(processed_files)} processed files")
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
            processed_files = set()
    
    # Find XML files
    xml_files = list(input_dir.glob("**/*.xml"))
    logger.info(f"Found {len(xml_files)} XML files")
    
    # Apply sampling if specified
    if args.sample_rate < 1.0:
        import random
        sample_size = int(len(xml_files) * args.sample_rate)
        xml_files = random.sample(xml_files, sample_size)
        logger.info(f"Sampling {len(xml_files)} files ({args.sample_rate:.1%})")
    
    # Apply max files limit
    if args.max_files:
        xml_files = xml_files[:args.max_files]
        logger.info(f"Limited to {len(xml_files)} files")
    
    # Filter out already processed files
    if processed_files:
        xml_files = [f for f in xml_files if f.name not in processed_files]
        logger.info(f"After filtering processed files: {len(xml_files)} remaining")
    
    if not xml_files:
        logger.info("No files to process")
        return
    
    # Validation-only mode
    if args.validate_only:
        logger.info("Running validation-only mode")
        processor = ProductionXMLProcessor(config)
        valid_files = 0
        for xml_file in xml_files:
            if processor.validate_xml_file(xml_file):
                valid_files += 1
        logger.info(f"Validation complete: {valid_files}/{len(xml_files)} files valid")
        return
    
    # Create batches
    batches = []
    for i in range(0, len(xml_files), args.batch_size):
        batch_files = xml_files[i:i + args.batch_size]
        batch_id = i // args.batch_size
        batches.append((batch_files, batch_id, config))
    
    logger.info(f"Created {len(batches)} batches for processing")
    
    # Process batches
    total_stats = {
        'total_files': len(xml_files),
        'processed_files': 0,
        'failed_files': 0,
        'skipped_files': 0,
        'examples_generated': 0,
        'processing_errors': [],
        'batch_results': []
    }
    
    start_time = time.time()
    
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(process_file_batch, batch): batch[1] 
                for batch in batches
            }
            
            # Process results as they complete
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                
                try:
                    batch_stats, returned_batch_id, examples = future.result(timeout=300)  # 5 minute timeout
                    
                    # Write examples immediately
                    if examples:
                        component_counts = file_writer.write_examples_to_file(
                            examples, batch_id, output_dir
                        )
                        batch_stats['component_counts'] = component_counts
                    
                    # Update total statistics
                    for key in ['processed_files', 'failed_files', 'skipped_files', 'examples_generated']:
                        total_stats[key] += batch_stats.get(key, 0)
                    
                    total_stats['processing_errors'].extend(batch_stats.get('processing_errors', []))
                    total_stats['batch_results'].append({
                        'batch_id': batch_id,
                        'stats': batch_stats
                    })
                    
                    # Update progress
                    batch_files = batches[batch_id][0]
                    processed_files.update(f.name for f in batch_files)
                    
                    # Save progress periodically
                    if completed_batches % 10 == 0:  # Every 10 batches
                        try:
                            with open(checkpoint_file, 'w') as f:
                                json.dump(list(processed_files), f)
                        except Exception as e:
                            logger.warning(f"Failed to save progress: {e}")
                    
                    completed_batches += 1
                    progress_percent = (completed_batches / len(batches)) * 100
                    logger.info(f"Progress: {completed_batches}/{len(batches)} batches ({progress_percent:.1f}%) - "
                              f"Batch {batch_id}: {batch_stats['processed_files']}/{batch_stats['total_files']} files")
                    
                except Exception as e:
                    logger.error(f"Batch {batch_id} failed: {e}")
                    logger.debug(traceback.format_exc())
                    continue
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.debug(traceback.format_exc())
        return
    
    # Merge all batch files
    components = ['chatbot', 'predictor', 'explainer']
    merge_batch_files(output_dir, components, config)
    
    # Calculate final statistics
    end_time = time.time()
    processing_time = end_time - start_time
    
    total_stats.update({
        'processing_time_seconds': processing_time,
        'processing_time_formatted': f"{processing_time/60:.1f} minutes",
        'success_rate': total_stats['processed_files'] / total_stats['total_files'] if total_stats['total_files'] > 0 else 0,
        'files_per_second': total_stats['processed_files'] / processing_time if processing_time > 0 else 0,
        'examples_per_file': total_stats['examples_generated'] / max(1, total_stats['processed_files'])
    })
    
    # Save comprehensive report
    save_processing_report(total_stats, output_dir, config)
    
    # Clean up
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
        except:
            pass
    
    # Print final summary
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE!")
    logger.info(f"Processed: {total_stats['processed_files']}/{total_stats['total_files']} files")
    logger.info(f"Success rate: {total_stats['success_rate']:.1%}")
    logger.info(f"Processing time: {total_stats['processing_time_formatted']}")
    logger.info(f"Examples generated: {total_stats['examples_generated']} "
               f"({total_stats['examples_per_file']:.1f} per file)")
    logger.info(f"Processing speed: {total_stats['files_per_second']:.1f} files/second")
    
    if total_stats['processing_errors']:
        logger.warning(f"Errors encountered: {len(total_stats['processing_errors'])}")
        error_types = Counter(error.get('type', 'unknown') for error in total_stats['processing_errors'])
        for error_type, count in error_types.most_common():
            logger.warning(f"  {error_type}: {count}")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    main()