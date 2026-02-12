#!/usr/bin/env python3
"""
filter.py - Apply aggressive filtering to raw samples

This is where most samples die. That's good - we only want the best.

Usage:
    python filter.py --input raw_data.jsonl --output filtered_data.jsonl
    python filter.py --input raw_data.jsonl --output filtered_data.jsonl --strict
"""

import json
import argparse
import re
from collections import defaultdict, Counter
from typing import Optional

# ============================================================================
# FILTER CONFIGURATIONS
# ============================================================================

FILTER_CONFIG = {
    "normal": {
        "min_question_length": 50,
        "max_question_length": 500,
        "min_reasoning_steps": 2,
        "max_reasoning_steps": 5,
        "min_step_length": 15,
        "consistency_threshold": 0.66,  # 2 out of 3 must agree
    },
    "strict": {
        "min_question_length": 70,
        "max_question_length": 400,
        "min_reasoning_steps": 2,
        "max_reasoning_steps": 5,
        "min_step_length": 20,
        "consistency_threshold": 1.0,  # All must agree
    }
}

# ============================================================================
# FILTER 1: RULE-BASED (FAST)
# ============================================================================

class RuleFilter:
    """Fast deterministic checks - fail early."""
    
    def __init__(self, config: dict):
        self.config = config
        self.failures = defaultdict(int)
    
    def check(self, sample: dict) -> tuple[bool, Optional[str]]:
        """Return (passed, failure_reason)."""
        
        # Check 1: Question length
        q_len = len(sample["question"])
        if q_len < self.config["min_question_length"]:
            return False, f"question_too_short_{q_len}"
        if q_len > self.config["max_question_length"]:
            return False, f"question_too_long_{q_len}"
        
        # Check 2: Number of reasoning steps
        num_steps = len(sample["reasoning"])
        if num_steps < self.config["min_reasoning_steps"]:
            return False, f"too_few_steps_{num_steps}"
        if num_steps > self.config["max_reasoning_steps"]:
            return False, f"too_many_steps_{num_steps}"
        
        # Check 3: Step count matches metadata
        if sample.get("num_steps") != num_steps:
            return False, f"step_count_mismatch"
        
        # Check 4: Each step has substance
        for i, step in enumerate(sample["reasoning"]):
            if len(step.strip()) < self.config["min_step_length"]:
                return False, f"step_{i+1}_too_short"
        
        # Check 5: Answer has numerical component
        if not re.search(r'\d+\.?\d*', sample["answer"]):
            return False, "answer_not_numerical"
        
        # Check 6: No prohibited patterns (AI refusals, apologies)
        prohibited = ["i cannot", "i apologize", "as an ai", "i'm sorry"]
        combined_text = (
            sample["question"] + " " + 
            " ".join(sample["reasoning"]) + " " + 
            sample["answer"]
        ).lower()
        
        for pattern in prohibited:
            if pattern in combined_text:
                return False, f"prohibited_pattern_{pattern.replace(' ', '_')}"
        
        # Check 7: Reasoning steps are distinct (not just repeated)
        if len(set(sample["reasoning"])) < len(sample["reasoning"]) * 0.8:
            return False, "repeated_steps"
        
        return True, None
    
    def filter_batch(self, samples: list[dict]) -> tuple[list[dict], dict]:
        """Filter batch and return (passed, failure_stats)."""
        passed = []
        
        for sample in samples:
            is_valid, reason = self.check(sample)
            
            if is_valid:
                passed.append(sample)
            else:
                self.failures[reason] += 1
        
        return passed, dict(self.failures)

# ============================================================================
# FILTER 2: CONSISTENCY CHECK
# ============================================================================

class ConsistencyFilter:
    """Same question → same answer? If not, something's wrong."""
    
    def __init__(self, config: dict):
        self.threshold = config["consistency_threshold"]
        self.inconsistent_groups = []
    
    def normalize_question(self, question: str) -> str:
        """Normalize question for grouping."""
        normalized = " ".join(question.split())
        normalized = normalized.lower()
        normalized = normalized.rstrip("?!.")
        # Remove specific numbers to group similar structures
        # normalized = re.sub(r'\d+\.?\d*', 'N', normalized)
        return normalized
    
    def extract_numerical_answer(self, answer: str) -> Optional[float]:
        """Extract number from answer."""
        match = re.search(r'(\d+\.?\d*)', answer)
        if match:
            try:
                return float(match.group(1))
            except:
                return None
        return None
    
    def check_consistency(self, samples: list[dict]) -> list[dict]:
        """Group by question, check if answers agree."""
        
        # Group by normalized question
        groups = defaultdict(list)
        for sample in samples:
            key = self.normalize_question(sample["question"])
            groups[key].append(sample)
        
        consistent = []
        
        for question_key, group in groups.items():
            # Single sample - can't verify, but include
            if len(group) == 1:
                sample = group[0]
                sample["consistency_verified"] = False
                sample["agreement_count"] = 1
                consistent.append(sample)
                continue
            
            # Multiple samples - check agreement
            answers = [self.extract_numerical_answer(s["answer"]) for s in group]
            valid_answers = [a for a in answers if a is not None]
            
            if not valid_answers:
                # Can't check consistency
                continue
            
            # Calculate agreement
            mean_answer = sum(valid_answers) / len(valid_answers)
            tolerance = 0.05  # 5% tolerance
            
            agreeing = sum(
                1 for ans in valid_answers
                if abs(ans - mean_answer) / max(mean_answer, 0.01) <= tolerance
            )
            
            agreement_rate = agreeing / len(valid_answers)
            
            if agreement_rate >= self.threshold:
                # Pick the best sample from the group
                best = self._select_best(group)
                best["consistency_verified"] = True
                best["agreement_count"] = len(group)
                best["agreement_rate"] = agreement_rate
                consistent.append(best)
            else:
                # Inconsistent answers - reject all
                self.inconsistent_groups.append({
                    "question": question_key[:100],
                    "answers": [s["answer"] for s in group],
                    "count": len(group)
                })
        
        return consistent
    
    def _select_best(self, samples: list[dict]) -> dict:
        """Select highest quality sample from group."""
        # Score by reasoning detail
        scored = []
        for sample in samples:
            score = 0
            
            # Longer reasoning is often better
            avg_step_len = sum(len(s) for s in sample["reasoning"]) / len(sample["reasoning"])
            score += min(avg_step_len / 50, 10)
            
            # Explicit step labels
            step_labels = sum(1 for s in sample["reasoning"] if "step" in s.lower())
            score += step_labels * 2
            
            # Prefer detailed variant
            if sample.get("prompt_variant") == "detailed":
                score += 3
            
            scored.append((score, sample))
        
        scored.sort(reverse=True)
        return scored[0][1]
    
    def get_stats(self) -> dict:
        """Get consistency statistics."""
        return {
            "inconsistent_groups": len(self.inconsistent_groups),
            "examples": self.inconsistent_groups[:5]
        }

# ============================================================================
# FILTER 3: QUALITY HEURISTICS
# ============================================================================

class QualityFilter:
    """Heuristic quality checks - catches subtle issues."""
    
    def __init__(self):
        self.failures = defaultdict(int)
    
    def check(self, sample: dict) -> tuple[bool, Optional[str]]:
        """Check quality indicators."""
        
        # Check 1: Reasoning should show work, not just state answers
        calculation_indicators = ["+", "-", "×", "÷", "=", "/", "*"]
        has_calculations = any(
            any(ind in step for ind in calculation_indicators)
            for step in sample["reasoning"]
        )
        
        if not has_calculations:
            return False, "no_calculations_shown"
        
        # Check 2: Steps should build on each other
        # Look for connecting words
        connecting_words = ["then", "next", "finally", "therefore", "so", "thus"]
        has_flow = any(
            any(word in step.lower() for word in connecting_words)
            for step in sample["reasoning"][1:]  # Skip first step
        )
        
        # This is optional - don't fail if missing
        # if not has_flow:
        #     return False, "no_logical_flow"
        
        # Check 3: Answer should match last step
        last_step = sample["reasoning"][-1]
        answer_num = re.search(r'(\d+\.?\d*)', sample["answer"])
        
        if answer_num:
            answer_value = answer_num.group(1)
            if answer_value not in last_step:
                # Could be calculated differently, not a hard fail
                pass
        
        # Check 4: Question should be clear and specific
        vague_words = ["something", "some things", "stuff", "do this", "figure out"]
        if any(word in sample["question"].lower() for word in vague_words):
            return False, "vague_question"
        
        return True, None
    
    def filter_batch(self, samples: list[dict]) -> tuple[list[dict], dict]:
        """Filter batch and return (passed, failure_stats)."""
        passed = []
        
        for sample in samples:
            is_valid, reason = self.check(sample)
            
            if is_valid:
                passed.append(sample)
            else:
                self.failures[reason] += 1
        
        return passed, dict(self.failures)

# ============================================================================
# MAIN FILTERING PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Filter raw training samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal filtering (expect 40-60% retention)
  python filter.py --input raw.jsonl --output filtered.jsonl
  
  # Strict filtering (expect 20-40% retention)
  python filter.py --input raw.jsonl --output filtered.jsonl --strict
  
  # Show stats only
  python filter.py --input raw.jsonl --stats-only
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file (raw data)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output JSONL file (filtered data)"
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict filtering rules"
    )
    
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't save output"
    )
    
    args = parser.parse_args()
    
    if not args.stats_only and not args.output:
        parser.error("--output required unless --stats-only is used")
    
    # Load data
    print("="*70)
    print("FILTERING PIPELINE")
    print("="*70)
    print(f"Loading from: {args.input}")
    
    samples = []
    with open(args.input, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    initial_count = len(samples)
    print(f"Loaded: {initial_count} samples")
    
    # Select config
    config = FILTER_CONFIG["strict" if args.strict else "normal"]
    mode = "STRICT" if args.strict else "NORMAL"
    print(f"Mode: {mode}")
    
    # Filter 1: Rule-based
    print(f"\n{'='*70}")
    print("FILTER 1: RULE-BASED CHECKS")
    print(f"{'='*70}")
    
    rule_filter = RuleFilter(config)
    samples, rule_failures = rule_filter.filter_batch(samples)
    
    print(f"Passed: {len(samples)}/{initial_count} ({len(samples)/initial_count*100:.1f}%)")
    
    if rule_failures:
        print(f"\nTop failure reasons:")
        for reason, count in sorted(rule_failures.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {reason:30s}: {count:4d}")
    
    # Filter 2: Consistency
    print(f"\n{'='*70}")
    print("FILTER 2: CONSISTENCY CHECK")
    print(f"{'='*70}")
    
    before_consistency = len(samples)
    consistency_filter = ConsistencyFilter(config)
    samples = consistency_filter.check_consistency(samples)
    
    print(f"Passed: {len(samples)}/{before_consistency} ({len(samples)/before_consistency*100:.1f}%)")
    
    consistency_stats = consistency_filter.get_stats()
    if consistency_stats["inconsistent_groups"] > 0:
        print(f"\nInconsistent groups found: {consistency_stats['inconsistent_groups']}")
        print(f"\nExamples of inconsistent answers:")
        for ex in consistency_stats["examples"][:3]:
            print(f"  Q: {ex['question'][:60]}...")
            print(f"  Conflicting answers: {ex['answers']}")
    
    # Filter 3: Quality heuristics
    print(f"\n{'='*70}")
    print("FILTER 3: QUALITY HEURISTICS")
    print(f"{'='*70}")
    
    before_quality = len(samples)
    quality_filter = QualityFilter()
    samples, quality_failures = quality_filter.filter_batch(samples)
    
    print(f"Passed: {len(samples)}/{before_quality} ({len(samples)/before_quality*100:.1f}%)")
    
    if quality_failures:
        print(f"\nQuality failure reasons:")
        for reason, count in sorted(quality_failures.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason:30s}: {count:4d}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FILTERING SUMMARY")
    print(f"{'='*70}")
    print(f"Initial samples:     {initial_count:5d}")
    print(f"After rules:         {before_consistency:5d} ({before_consistency/initial_count*100:5.1f}%)")
    print(f"After consistency:   {before_quality:5d} ({before_quality/initial_count*100:5.1f}%)")
    print(f"After quality:       {len(samples):5d} ({len(samples)/initial_count*100:5.1f}%)")
    print(f"\n{'─'*70}")
    print(f"FINAL RETENTION:     {len(samples)/initial_count*100:.1f}%")
    print(f"{'─'*70}")
    
    # Dataset statistics
    print(f"\nFinal dataset distribution:")
    
    difficulties = Counter(s["difficulty"] for s in samples)
    print(f"\nBy difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff:8s}: {count:4d}")
    
    domains = Counter(s["domain"] for s in samples)
    print(f"\nBy domain:")
    for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain:25s}: {count:4d}")
    
    verified = sum(1 for s in samples if s.get("consistency_verified", False))
    print(f"\nConsistency verified: {verified}/{len(samples)} ({verified/len(samples)*100:.1f}%)")
    
    # Save
    if not args.stats_only:
        print(f"\n💾 Saving to: {args.output}")
        
        with open(args.output, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"✓ Saved {len(samples)} samples")
    
    # Retention expectations
    print(f"\n{'='*70}")
    print("RETENTION ANALYSIS")
    print(f"{'='*70}")
    
    retention = len(samples) / initial_count * 100
    
    if retention < 30:
        print("⚠️  LOW RETENTION (<30%)")
        print("   Filters may be too strict, or generation quality is poor.")
        print("   Check failure reasons above and adjust accordingly.")
    elif retention > 70:
        print("⚠️  HIGH RETENTION (>70%)")
        print("   Filters may be too lenient.")
        print("   Consider using --strict mode or adding domain-specific rules.")
    else:
        print("✓ HEALTHY RETENTION (30-70%)")
        print("  This is expected. Aggressive filtering = high quality.")

if __name__ == "__main__":
    main()
