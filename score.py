#!/usr/bin/env python3
"""
score.py - Score samples using model-as-judge

This is expensive but catches subtle quality issues that rules miss.

Usage:
    python score.py --input filtered_data.jsonl --output scored_data.jsonl --threshold 7.0
    python score.py --input filtered_data.jsonl --stats-only
"""

import json
import argparse
import os
from pathlib import Path
from typing import Optional
from collections import Counter

from local_llm import call_local_llm, with_model_override

# ============================================================================
# SCORING PROMPT
# ============================================================================

SCORING_PROMPT = """You are evaluating the quality of step-by-step mathematical reasoning.

Problem:
{question}

Reasoning Steps:
{reasoning}

Final Answer:
{answer}

Evaluate this solution on a scale of 1-10 based on:
1. **Logical flow**: Are steps in the right order?
2. **Clarity**: Is each step clearly explained?
3. **Completeness**: Are calculations shown explicitly?
4. **Correctness**: Does reasoning lead to the stated answer?
5. **Format**: Is the answer properly formatted with units?

Return ONLY a JSON object:
{{
  "score": <number 1-10>,
  "reasoning": "<brief explanation of score>",
  "issues": ["<issue 1>", "<issue 2>", ...]
}}

Scoring guidelines:
- 1-3: Fundamentally flawed (wrong logic, missing steps)
- 4-6: Correct but unclear or incomplete
- 7-8: Good quality, minor issues only
- 9-10: Exceptional clarity and completeness

Be strict. Most samples should score 5-7."""

# ============================================================================
# LLM INTEGRATION
# ============================================================================

def call_llm_for_scoring(prompt: str) -> str:
    """
    Delegate scoring queries to the local LLM helper (Ollama llama2 by default).
    """

    score_model = os.environ.get("SCORE_LLM_MODEL")
    if score_model:
        with with_model_override(score_model):
            return call_local_llm(prompt, max_new_tokens=600, temperature=0.3, top_p=0.9)
    return call_local_llm(prompt, max_new_tokens=600, temperature=0.3, top_p=0.9)

# ============================================================================
# SCORING LOGIC
# ============================================================================

def parse_score_response(response: str) -> Optional[dict]:
    """Parse LLM scoring response."""
    try:
        # Remove markdown
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        parsed = json.loads(response.strip())
        
        # Validate
        if "score" in parsed and "reasoning" in parsed:
            return parsed
        
        return None
    except:
        return None

def score_sample(sample: dict) -> Optional[dict]:
    """Score a single sample using LLM."""
    
    # Format prompt
    prompt = SCORING_PROMPT.format(
        question=sample["question"],
        reasoning="\n".join(f"{i+1}. {step}" for i, step in enumerate(sample["reasoning"])),
        answer=sample["answer"]
    )
    
    # Try up to 2 times
    for attempt in range(2):
        try:
            response = call_llm_for_scoring(prompt)
            score_data = parse_score_response(response)
            
            if score_data:
                # Add scores to sample
                return {
                    **sample,
                    "quality_score": score_data["score"],
                    "score_reasoning": score_data["reasoning"],
                    "score_issues": score_data.get("issues", [])
                }
        except Exception as e:
            print(f"  Scoring attempt {attempt + 1} failed: {e}")
            if attempt < 1:
                import time
                time.sleep(1)
    
    return None

def score_batch(samples: list[dict], show_progress: bool = True) -> list[dict]:
    """Score a batch of samples."""
    scored = []
    failed = 0
    
    for i, sample in enumerate(samples):
        if show_progress and (i + 1) % 10 == 0:
            print(f"  Scored {i+1}/{len(samples)}...")
        
        scored_sample = score_sample(sample)
        
        if scored_sample:
            scored.append(scored_sample)
        else:
            failed += 1
    
    if failed > 0:
        print(f"  ⚠️  Failed to score {failed} samples")
    
    return scored

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score samples using model-as-judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score all samples, keep those >= 7.0
  python score.py --input filtered.jsonl --output scored.jsonl --threshold 7.0
  
  # Score and show statistics only
  python score.py --input filtered.jsonl --stats-only
  
  # Score with different threshold
  python score.py --input filtered.jsonl --output high_quality.jsonl --threshold 8.0
  
Note: This script runs the local LLM for each sample; inference can be slow on CPU or without a GPU.
      Consider sampling first: head -n 100 filtered.jsonl > sample.jsonl
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file (filtered data)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output JSONL file (scored data)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=6.0,
        help="Minimum score to keep (default: 6.0)"
    )
    
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't save output"
    )
    
    parser.add_argument(
        "--sample", "-n",
        type=int,
        help="Only score first N samples (for testing)"
    )
    
    args = parser.parse_args()
    
    if (
        not args.stats_only
        and not args.output
        and str(args.input).lower() != "latest"
        and not Path(args.input).is_dir()
    ):
        parser.error("--output required unless --stats-only is used")
    
    # Load data
    print("="*70)
    print("QUALITY SCORING")
    print("="*70)
    print(f"Loading from: {args.input}")

    input_path = Path(args.input)
    if str(args.input).lower() == "latest":
        runs_dir = Path("runs")
        if not runs_dir.exists():
            print("Error: runs/ does not exist. Run generate.py first.")
            return
        run_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
        if not run_dirs:
            print("Error: No run_* directories found under runs/.")
            return
        latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
        input_path = latest
        print(f"Using latest run directory: {input_path}")

    if input_path.is_dir():
        filtered_files = list(input_path.rglob("filtered_*.jsonl"))
        if not filtered_files:
            print(f"Error: No filtered_*.jsonl files found in {input_path}")
            return
        for filtered_file in filtered_files:
            output_file = filtered_file.with_name(filtered_file.name.replace("filtered_", "final_"))
            print(f"\nProcessing: {filtered_file} -> {output_file}")
            _run_score_single(filtered_file, output_file, args.threshold, args.stats_only, args.sample)
        return

    output_file = Path(args.output) if args.output else None
    if output_file is None and args.stats_only:
        output_file = Path("scored.jsonl")
    _run_score_single(input_path, output_file, args.threshold, args.stats_only, args.sample)

if __name__ == "__main__":
    main()


def _run_score_single(
    input_file: Path,
    output_file: Optional[Path],
    threshold: float,
    stats_only: bool,
    sample: Optional[int],
) -> None:
    print("=" * 70)
    print("QUALITY SCORING")
    print("=" * 70)
    print(f"Loading from: {input_file}")

    samples = []
    with open(input_file, "r") as f:
        for line in f:
            samples.append(json.loads(line))

    if sample:
        samples = samples[:sample]
        print(f"Sampling: {len(samples)} samples (for testing)")

    initial_count = len(samples)
    print(f"Loaded: {initial_count} samples")
    print(f"Threshold: {threshold}")

    print(f"\n{'='*70}")
    print("SCORING SAMPLES")
    print(f"{'='*70}")
    print(f"⚠️  This runs the local LLM for each sample and can be slow")
    print(f"   Estimated calls: {len(samples)}")

    scored_samples = score_batch(samples)

    print(f"\nSuccessfully scored: {len(scored_samples)}/{initial_count}")

    passed = [s for s in scored_samples if s["quality_score"] >= threshold]

    print(f"\n{'='*70}")
    print("SCORING STATISTICS")
    print(f"{'='*70}")

    if scored_samples:
        scores = [s["quality_score"] for s in scored_samples]

        print(f"\nScore distribution:")
        print(f"  Mean:   {sum(scores)/len(scores):.2f}")
        print(f"  Median: {sorted(scores)[len(scores)//2]:.2f}")
        print(f"  Min:    {min(scores):.2f}")
        print(f"  Max:    {max(scores):.2f}")

        print(f"\nScore histogram:")
        score_bins = Counter()
        for score in scores:
            bin = int(score)
            score_bins[bin] += 1

        for bin in sorted(score_bins.keys()):
            count = score_bins[bin]
            pct = count / len(scores) * 100
            bar = "█" * int(pct / 2)
            print(f"  {bin:2d}: {bar} {count:4d} ({pct:5.1f}%)")

        print(f"\n{'─'*70}")
        print(f"Above threshold ({threshold}): {len(passed)}/{len(scored_samples)} ({len(passed)/len(scored_samples)*100:.1f}%)")
        print(f"{'─'*70}")

    if scored_samples:
        print(f"\n{'='*70}")
        print("COMMON ISSUES")
        print(f"{'='*70}")

        all_issues = []
        for s in scored_samples:
            all_issues.extend(s.get("score_issues", []))

        if all_issues:
            issue_counts = Counter(all_issues)
            print(f"\nMost common issues:")
            for issue, count in issue_counts.most_common(10):
                print(f"  • {issue}: {count}")
        else:
            print("No issues reported")

    print(f"\n{'='*70}")
    print("EXAMPLES")
    print(f"{'='*70}")

    if scored_samples:
        high_score = max(scored_samples, key=lambda s: s["quality_score"])
        print(f"\nHighest scoring ({high_score['quality_score']:.1f}):")
        print(f"Q: {high_score['question'][:80]}...")
        print(f"Reasoning: {high_score['score_reasoning']}")

        low_score = min(scored_samples, key=lambda s: s["quality_score"])
        print(f"\nLowest scoring ({low_score['quality_score']:.1f}):")
        print(f"Q: {low_score['question'][:80]}...")
        print(f"Reasoning: {low_score['score_reasoning']}")
        if low_score.get("score_issues"):
            print(f"Issues: {', '.join(low_score['score_issues'])}")

    if not stats_only:
        if output_file is None:
            raise ValueError("output_file is required unless stats_only is True")
        print(f"\n{'='*70}")
        print(f"SAVING RESULTS")
        print(f"{'='*70}")
        print(f"Saving {len(passed)} samples to: {output_file}")

        with open(output_file, "w") as f:
            for sample in passed:
                f.write(json.dumps(sample) + "\n")

        print(f"✓ Saved {len(passed)} high-quality samples")

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Started with:        {initial_count:5d} samples")
    print(f"Successfully scored: {len(scored_samples):5d} samples")
    final_percent = (len(passed) / initial_count * 100) if initial_count > 0 else 0.0
    print(f"Above threshold:     {len(passed):5d} samples ({final_percent:.1f}%)")

    if not stats_only:
        print(f"\n✨ High-quality dataset ready: {output_file}")
