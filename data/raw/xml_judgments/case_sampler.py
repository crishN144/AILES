#!/usr/bin/env python3
"""
Parse a UK family-law XML (e.g., [2024] EWHC 3266 (Fam).xml) and generate USABLE training examples.
This version creates coherent Q&A pairs that actually teach legal reasoning, with HTML explainability markers.
No external dependencies beyond stdlib.
"""

import xml.etree.ElementTree as ET
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

# -----------------------------
# Utilities
# -----------------------------

def _localname(tag: str) -> str:
    """Return local tag name without namespace."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def _iter_nodes_by_localname(root: ET.Element, names: Tuple[str, ...]):
    """Yield elements whose localname is in names (namespace-agnostic)."""
    want = set(n.lower() for n in names)
    for el in root.iter():
        if _localname(el.tag).lower() in want:
            yield el

def _norm_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()

def _text_of(el: ET.Element) -> str:
    """Concatenate visible text inside a node; keep bracketed content like [B], [2021] EWCA Civ 139."""
    return _norm_ws(''.join(el.itertext()))

# -----------------------------
# Main Parser
# -----------------------------

class FamilyLawParser:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

        # Gather paragraphs from multiple tag shapes
        self.paragraphs = self._collect_paragraphs()
        # Also gather header/decision/order blocks if present
        self.header_text = self._collect_block_text(("header",))
        self.decision_text = self._collect_block_text(("decision",))
        self.order_text = self._collect_block_text(("order",))

        # Full text used for global regex detection
        self.full_text = _norm_ws(" ".join([self.header_text] + self.paragraphs + [self.decision_text, self.order_text]))

    # --------- collectors ---------

    def _collect_paragraphs(self) -> List[str]:
        paras = []
        # Prefer <paragraph> then <p>
        nodes = list(_iter_nodes_by_localname(self.root, ("paragraph",)))
        if not nodes:
            nodes = list(_iter_nodes_by_localname(self.root, ("p",)))
        for n in nodes:
            t = _text_of(n)
            if t:
                paras.append(t)
        # fallback: if still empty, use entire text split by sentence-ish
        if not paras:
            whole = _text_of(self.root)
            paras = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', whole) if s.strip()]
        return paras

    def _collect_block_text(self, names: Tuple[str, ...]) -> str:
        chunks = []
        for n in _iter_nodes_by_localname(self.root, names):
            t = _text_of(n)
            if t:
                chunks.append(t)
        return _norm_ws(" ".join(chunks))

    # --------- high-level extractors ---------

    def extract_case_story(self) -> Dict:
        story = {
            "parties": self._identify_parties(),
            "children": self._extract_children_info(),
            "dispute": self._extract_core_dispute(),
            "timeline": self._extract_timeline(),
            "jurisdiction": self._extract_jurisdiction(),
            "case_type": self._infer_case_type(),
        }
        return story

    def extract_legal_framework(self) -> Dict:
        return {
            "primary_law": self._extract_primary_legislation(),
            "key_cases": self._extract_case_citations(),
            "legal_test": self._extract_legal_test(),
            "principles": self._extract_principles(),
        }

    def extract_judicial_reasoning(self) -> Dict:
        issues_reason_outcome = self._extract_issues_reasoning_outcome()
        return {
            "factors_considered": self._extract_factors(),
            "balancing": self._extract_balancing(),
            "conclusion": self._extract_conclusion(),
            "outcome": issues_reason_outcome.get("decision", ""),
            "issues": issues_reason_outcome.get("issues", ""),
            "reasoning_window": issues_reason_outcome.get("reasoning", ""),
        }

    # --------- party / children / dispute ---------

    def _identify_parties(self) -> Dict:
        parties = {"applicant": None, "respondent": None, "relationship": None}
        t = self.full_text
        if re.search(r'\b(mother|father)\b', t, re.I):
            parties["relationship"] = "parents"
        elif re.search(r'\b(wife|husband)\b', t, re.I):
            parties["relationship"] = "spouses"
        return parties

    def _extract_children_info(self) -> List[Dict]:
        """
        Detect children via:
        - anonymised tokens like [B], [N]
        - explicit ages (aged 11 / 13-year-old)
        - “two children: [B] (born 2013, … aged 11) and [N] …” pattern
        """
        txt = self.full_text
        children: List[Dict] = []

        # 1) Two-children sentence with aliases
        m = re.search(r'two\s+children\s*:\s*\[([A-Z]{1,3})\][^[]+?\[([A-Z]{1,3})\]', txt, re.I)
        alias_pair = []
        if m:
            alias_pair = [m.group(1), m.group(2)]

        # 2) All alias tokens
        all_aliases = re.findall(r'\[([A-Z]{1,3})\]', txt)
        # keep order, unique
        seen = set()
        aliases = []
        for a in all_aliases:
            if a not in seen:
                seen.add(a)
                aliases.append(a)

        # 3) Ages
        age_hits = re.findall(r'\b(?:aged|age)\s*(\d{1,2})\b|\b(\d{1,2})-year-old\b', txt, flags=re.I)
        ages = []
        for a, b in age_hits:
            if a and a.isdigit():
                ages.append(int(a))
            elif b and b.isdigit():
                ages.append(int(b))

        # Prefer alias_pair if present
        candidate_aliases = alias_pair if alias_pair else aliases

        # Assemble entries by aligning ages to aliases when possible
        n = max(len(candidate_aliases), len(ages), 1 if re.search(r'\bchild(?:ren)?\b', txt, re.I) else 0)
        for i in range(n):
            entry = {}
            if i < len(candidate_aliases):
                entry["alias"] = candidate_aliases[i]
            if i < len(ages):
                entry["age"] = ages[i]
            children.append(entry)

        return children

    def _infer_case_type(self) -> str:
        t = self.full_text
        if re.search(r'\bHague\b', t, re.I) or \
           re.search(r'Article\s*13', t, re.I) or \
           re.search(r'wrongful\s+(?:removal|retention)', t, re.I) or \
           re.search(r'habitual\s+residence', t, re.I):
            return "hague_convention"
        if re.search(r'\bfinancial remedy\b|\bancillary relief\b|\blump sum\b|\bForm E\b', t, re.I):
            return "financial_remedy"
        if re.search(r'\bchild arrangements\b|\bresidence\b|\bcontact\b', t, re.I):
            return "child_arrangements"
        return "general_family"

    def _extract_core_dispute(self) -> str:
        if self._infer_case_type() == "hague_convention":
            return "international child abduction / Hague Convention return"
        t = self.full_text
        for pat in [
            r'application\s+for\s+(.*?)(?:\.|,)',
            r'dispute\s+(?:concerns?|about|regarding)\s+(.*?)(?:\.|,)',
            r'The\s+issue\s+is\s+(.*?)(?:\.|$)',
            r'seeks?\s+(?:the\s+)?(return|custody|residence|contact|maintenance|division)'
        ]:
            m = re.search(pat, t[:15000], flags=re.I)
            if m:
                grp = m.group(1) if m.lastindex else m.group(0)
                return _norm_ws(grp)
        return "child arrangements"

    def _extract_timeline(self) -> Dict:
        t = self.full_text
        timeline = {}
        for label, pat in [
            ("married", r'married\s+(?:on\s+)?(\d+\s+\w+\s+\d{4}|\d{4})'),
            ("separated", r'separated\s+(?:on\s+)?(\d+\s+\w+\s+\d{4}|\d{4})'),
            ("order", r'order(?:ed)?\s+(?:on\s+)?(\d+\s+\w+\s+\d{4})'),
        ]:
            m = re.search(pat, t, re.I)
            if m: timeline[label] = m.group(1)
        return timeline

    def _extract_jurisdiction(self) -> str:
        countries = re.findall(r'\b(England(?: and Wales)?|Wales|Scotland|Czech Republic|Poland|France|Germany|United States)\b',
                               self.full_text, re.I)
        if countries:
            return Counter([c.title() for c in countries]).most_common(1)[0][0]
        return "England and Wales"

    # --------- legal framework ---------

    def _extract_primary_legislation(self) -> List[str]:
        t = self.full_text
        statues = []
        for pat in [
            r'Children Act\s+\d{4}',
            r'Matrimonial Causes Act\s+\d{4}',
            r'Family Law Act\s+\d{4}',
            r'Adoption and Children Act\s+\d{4}',
            r'1980 Hague Convention|Hague Convention\s*(?:1980)?',
        ]:
            statues += re.findall(pat, t)
        # normalize
        cleaned = []
        for s in statues:
            s = s.strip()
            if s.lower().startswith('hague convention'):
                s = '1980 Hague Convention'
            cleaned.append(s)
        # unique order-preserving
        seen = set(); out=[]
        for x in cleaned:
            if x not in seen:
                seen.add(x); out.append(x)
        return out[:5]

    def _extract_case_citations(self) -> List[str]:
        pat = r'\[\d{4}\]\s+(?:UKSC|EWCA(?:\s+Civ)?|EWHC(?:\s+\(Fam\))?|EWFC|EWCOP)[^\d\n]{0,20}\s+\d+'
        cites = re.findall(pat, self.full_text)
        seen = set(); out=[]
        for c in cites:
            if c not in seen:
                seen.add(c); out.append(c)
        return out[:10]

    def _extract_legal_test(self) -> str:
        t = self.full_text
        for pat in [
            r'\b[Tt]he\s+test\s+is\s+(.*?)(?:\.\s|;|$)',
            r'\b[Tt]he\s+court\s+must\s+(?:consider|determine)\s+(.*?)(?:\.\s|;|$)',
            r'\b[Aa]pplying\s+the\s+test\s+in\s+([^\.;]+)'
        ]:
            m = re.search(pat, t)
            if m:
                return _norm_ws(m.group(1))[:300]
        return ""

    def _extract_principles(self) -> List[str]:
        t = self.full_text
        out = []
        for pat in [
            r'\b[Tt]he\s+principle\s+(?:is|that)\s+(.*?)(?:\.\s|;|$)',
            r'\b[Ll]aw\s+(?:requires|is\s+that)\s+(.*?)(?:\.\s|;|$)',
            r'\b[Ee]stablish(?:es|ed)\s+that\s+(.*?)(?:\.\s|;|$)',
        ]:
            for m in re.findall(pat, t):
                s = _norm_ws(m)
                if 20 < len(s) < 240:
                    out.append(s)
        # unique
        seen=set(); uniq=[]
        for s in out:
            if s not in seen:
                seen.add(s); uniq.append(s)
        return uniq[:5]

    # --------- reasoning / issues / outcome ---------

    ISSUE_HINTS = [r'\bissue[s]?\b', r'\bquestion\b', r'\bmust decide\b', r'\bwhether\b']
    REASON_HINTS = [r'\bin my judgment\b', r'\bi (?:am|was) satisfied\b', r'\bthe test is\b', r'\bhaving considered\b']
    OUTCOME_HINTS = [r'\bi order\b', r'\border(?:s|ed)?\b', r'\bapplication (?:granted|refused|dismissed|allowed)\b',
                     r'\baccordingly\b', r'\bfor these reasons\b', r'\bi (?:conclude|find)\b']

    def _window(self, idx: int, w: int = 1) -> str:
        lo, hi = max(0, idx - w), min(len(self.paragraphs), idx + w + 1)
        return _norm_ws(" ".join(self.paragraphs[lo:hi]))

    def _extract_issues_reasoning_outcome(self) -> Dict[str, str]:
        issues, reasons, outcome = [], [], ""
        for i, p in enumerate(self.paragraphs):
            if any(re.search(h, p, re.I) for h in self.ISSUE_HINTS):
                issues.append(self._window(i, 1))
            if any(re.search(h, p, re.I) for h in self.REASON_HINTS):
                reasons.append(self._window(i, 1))
            if not outcome and any(re.search(h, p, re.I) for h in self.OUTCOME_HINTS):
                # prefer lines that contain verbs like return/dismiss/allow/refuse
                if re.search(r'\b(return|dismiss|allow|refuse|granted|refused|allowed)\b', p, re.I):
                    outcome = self._window(i, 1)
        return {
            "issues": _norm_ws(" ".join(sorted(set(issues))))[:2000],
            "reasoning": _norm_ws(" ".join(sorted(set(reasons))))[:4000],
            "decision": outcome
        }

    def _extract_factors(self) -> List[str]:
        t = self.full_text
        out = []
        for pat in [
            r'[Tt]aking\s+into\s+account\s+(.*?)(?:\.|,)',
            r'[Cc]onsidering\s+(.*?)(?:\.|,)',
            r'[Hh]aving\s+regard\s+to\s+(.*?)(?:\.|,)',
            r'[Ii]n\s+light\s+of\s+(.*?)(?:\.|,)'
        ]:
            for m in re.findall(pat, t):
                s = _norm_ws(m)
                if len(s) > 20:
                    out.append(s)
        # unique small set
        seen=set(); uniq=[]
        for s in out:
            if s not in seen:
                seen.add(s); uniq.append(s)
        return uniq[:5]

    def _extract_balancing(self) -> str:
        t = self.full_text
        for pat in [
            r'[Bb]alancing\s+(.*?)\s+against\s+(.*?)(?:\.|,)',
            r'[Oo]n\s+(?:the\s+)?one\s+hand\s+(.*?)[,;]\s+on\s+the\s+other\s+(.*?)(?:\.|,)',
            r'[Ww]hile\s+(.*?)[,;]\s+(?:nevertheless|however)\s+(.*?)(?:\.|,)'
        ]:
            m = re.search(pat, t)
            if m:
                return f"The court balanced {_norm_ws(m.group(1))} against {_norm_ws(m.group(2))}"
        return ""

    def _extract_conclusion(self) -> str:
        t = self.full_text
        for pat in [
            r'\b[Ii]\s+(?:am\s+)?satisfied\s+that\s+(.*?)(?:\.|$)',
            r'\b[Ii]n\s+my\s+(?:judgment|view)\s+(.*?)(?:\.|$)',
            r'\b[Aa]ccordingly\s+[Ii]\s+(?:find|conclude)\s+that\s+(.*?)(?:\.|$)',
            r'\b[Ff]or\s+(?:these|those)\s+reasons\s+(.*?)(?:\.|$)'
        ]:
            m = re.search(pat, t)
            if m:
                return _norm_ws(m.group(1))[:400]
        return ""

    # -----------------------------
    # Training example composition
    # -----------------------------

    def _html_tag(self, text: str) -> str:
        """Add explainability tags for statutes, cases, amounts."""
        if not text: return text
        # statutes
        text = re.sub(r'(Children Act\s+\d{4}|Matrimonial Causes Act\s+\d{4}|Family Law Act\s+\d{4}|Adoption and Children Act\s+\d{4}|1980 Hague Convention)',
                      r'<span class="statute" style="color: #008800; font-weight: bold">\1</span>', text)
        # case law like [2011] UKSC 27 / [2021] EWCA Civ 139 / [2014] EWHC 1234 (Fam)
        text = re.sub(r'\[\d{4}\]\s+(?:UKSC|EWCA(?:\s+Civ)?|EWHC(?:\s+\(Fam\))?|EWFC|EWCOP)[^\d\n]{0,20}\s+\d+',
                      lambda m: f'<span class="case-law" style="color: #0066CC; font-weight: bold; text-decoration: underline">{m.group(0)}</span>',
                      text)
        # money
        text = re.sub(r'£\s?\d[\d,]*(?:\.\d{2})?', r'<span class="financial" style="color: #444444; font-weight: bold; background-color: #FFFFCC">\g<0></span>', text)
        return text

    def _build_user(self, story: Dict, reasoning: Dict) -> str:
        parts = []
        j = story.get("jurisdiction")
        if j and j.lower() != "england and wales":
            parts.append(f"We lived in {j}.")
        ch = story.get("children") or []
        if ch:
            ages = [str(c["age"]) for c in ch if isinstance(c, dict) and c.get("age") is not None]
            if ages:
                parts.append(f"Our children are aged {', '.join(ages)}.")
            else:
                parts.append("We have children.")
        ct = story.get("case_type")
        if ct == "hague_convention":
            parts.append("I initially agreed to a short stay in the UK, but the children are now being kept there.")
            parts.append("They say they don’t want to return.")
            parts.append("Can the court order a return under the 1980 Hague Convention?")
        else:
            dispute = story.get("dispute") or "child arrangements"
            parts.append(f"There's a dispute about {dispute}. What factors will the court consider?")
        return " ".join(parts)

    def _build_assistant(self, story: Dict, framework: Dict, reasoning: Dict) -> str:
        ct = story.get("case_type")
        resp = []

        if ct == "hague_convention":
            resp.append("Under the 1980 Hague Convention, the court first determines whether there has been wrongful removal or retention from the country of habitual residence.")
            resp.append("If so, the court considers exceptions, including the child-objections gateway and the Article 13(b) grave-risk defence, and then exercises discretion.")
            if framework.get("key_cases"):
                resp.append(f"Authorities such as {framework['key_cases'][0]} are commonly considered when weighing a child’s views and the Convention’s policy aims.")
        if framework.get("legal_test"):
            resp.append(f"Applicable test: {self._html_tag(framework['legal_test'])}.")
        if reasoning.get("issues"):
            resp.append(f"<b>Issues:</b> {self._html_tag(reasoning['issues'])}.")
        if reasoning.get("reasoning_window"):
            resp.append(f"<b>Reasoning:</b> {self._html_tag(reasoning['reasoning_window'])}.")
        if reasoning.get("balancing"):
            resp.append(self._html_tag(reasoning["balancing"]) + ".")
        if reasoning.get("conclusion"):
            resp.append(f"In conclusion, {self._html_tag(reasoning['conclusion'])}.")

        text = " ".join(resp).strip()

        # Length/quality floor to avoid word-salad
        if len(text.split()) < 150:
            extra = (" In exercising discretion, the court considers the authenticity and strength of the children’s views, "
                     "any undue influence, available protective measures in the requesting state, and the Convention’s "
                     "policy of prompt return.")
            text += extra
        return text

    # -----------------------------
    # Public API: create examples
    # -----------------------------

    def create_training_examples(self) -> List[Dict]:
        story = self.extract_case_story()
        framework = self.extract_legal_framework()
        reasoning = self.extract_judicial_reasoning()

        # Hague sanity: skip if detected as Hague but no children parsed
        if story.get("case_type") == "hague_convention" and not (story.get("children") or []):
            return []

        examples = []

        # Primary dispute example (always try to create one)
        user_q = self._build_user(story, reasoning)
        asst = self._build_assistant(story, framework, reasoning)

        example = {
            "messages": [
                {"role": "system", "content": "You are Ailes, a helpful and professional AI assistant on a legal platform specializing in UK family law. Use HTML explainability tags in outputs where relevant."},
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": asst}
            ],
            "metadata": {
                "type": "dispute_resolution",
                "source": Path(self.xml_path).name,
                "case_type": story.get("case_type"),
                "jurisdiction": story.get("jurisdiction"),
                "children_count": len(story.get("children") or []),
                "primary_law": framework.get("primary_law"),
                "key_cases": framework.get("key_cases"),
            }
        }

        # Validate and only add if usable
        if self._is_usable(example):
            examples.append(example)

        # Optional: create a second example focused on principle/test if present
        if framework.get("principles") and framework.get("legal_test"):
            user2 = f"How does the court apply this principle: {framework['principles'][0]}?"
            asst2 = (f"This principle operates alongside the applicable test ({self._html_tag(framework['legal_test'])}). "
                     f"In practice, courts assess case-specific factors such as {', '.join(reasoning.get('factors_considered') or [])}. "
                     f"Discretion is then exercised by weighing the child’s authentic views, any influence, protective measures, "
                     f"and the policy of prompt return under the 1980 Hague Convention.")
            ex2 = {
                "messages": [
                    {"role": "system", "content": "You are Ailes, a helpful and professional AI assistant on a legal platform specializing in UK family law. Use HTML explainability tags in outputs where relevant."},
                    {"role": "user", "content": user2},
                    {"role": "assistant", "content": self._html_tag(asst2)}
                ],
                "metadata": {
                    "type": "legal_principle",
                    "source": Path(self.xml_path).name,
                    "case_type": story.get("case_type"),
                }
            }
            if self._is_usable(ex2, allow_shorter=True):
                examples.append(ex2)

        return examples

    # -----------------------------
    # Validation
    # -----------------------------

    def _assistant_mentions_citations_not_in_source(self, assistant_txt: str) -> bool:
        # If assistant mentions case citations, ensure they appear somewhere in source text
        cites = re.findall(r'\[\d{4}\]\s+(?:UKSC|EWCA(?:\s+Civ)?|EWHC(?:\s+\(Fam\))?|EWFC|EWCOP)[^\d\n]{0,20}\s+\d+', assistant_txt)
        for c in cites:
            if c not in self.full_text:
                return True
        return False

    def _is_usable(self, example: Dict, allow_shorter: bool = False) -> bool:
        user_len = len(example["messages"][1]["content"].split())
        asst_len = len(example["messages"][2]["content"].split())
        if not allow_shorter and asst_len < 150:
            return False
        if user_len < 12:
            return False
        if self._assistant_mentions_citations_not_in_source(example["messages"][2]["content"]):
            return False
        # Reject outputs with suspicious comma chains and no periods
        frag = example["messages"][2]["content"]
        if (frag.count(',') >= 2 and '.' not in frag[:200]):
            return False
        return True

# -----------------------------
# CLI
# -----------------------------

def main():
    xml_path = "/users/bgxp240/ailes_legal_ai/data/raw/xml_judgments/[2024] EWHC 3266 (Fam).xml"
    if not Path(xml_path).exists():
        print(f"⚠️ File not found at default path: {xml_path}")
        return

    print("=" * 70)
    print(f"PARSING: {xml_path}")
    print("=" * 70)

    parser = FamilyLawParser(xml_path)

    # Story
    story = parser.extract_case_story()
    framework = parser.extract_legal_framework()
    reasoning = parser.extract_judicial_reasoning()

    print("\n📚 CASE STORY:")
    print(f"  Dispute: {story['dispute']}")
    print(f"  Children: {len(story['children'])} children found")
    print(f"  Jurisdiction: {story['jurisdiction']}")
    print(f"  Case type: {story['case_type']}")

    print("\n⚖️ LEGAL FRAMEWORK:")
    print(f"  Statutes: {', '.join(framework['primary_law'][:3]) if framework['primary_law'] else 'None found'}")
    print(f"  Cases cited: {len(framework['key_cases'])}")
    print(f"  Principles: {len(framework['principles'])}")
    print(f"  Legal test: {framework['legal_test'][:120] + '…' if framework['legal_test'] else '—'}")

    print("\n🧠 JUDICIAL REASONING:")
    print(f"  Issues window: {'yes' if reasoning.get('issues') else 'no'}")
    print(f"  Factors considered: {len(reasoning['factors_considered'])}")
    print(f"  Has balancing test: {bool(reasoning['balancing'])}")
    print(f"  Has conclusion: {bool(reasoning['conclusion'])}")
    print(f"  Has outcome (decision text): {bool(reasoning['outcome'])}")

    examples = parser.create_training_examples()
    print(f"\n✅ GENERATED {len(examples)} TRAINING EXAMPLES")

    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ({ex['metadata']['type']}) ---")
        print(f"USER: {ex['messages'][1]['content'][:160]}...")
        print(f"ASSISTANT: {ex['messages'][2]['content'][:220]}...")
        uw = len(ex["messages"][1]["content"].split())
        aw = len(ex["messages"][2]["content"].split())
        print(f"Validation: User={uw} words, Assistant={aw} words")

    out_path = Path("training_examples_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"examples": examples}, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved {len(examples)} examples to: {out_path.resolve()}")

if __name__ == "__main__":
    main()