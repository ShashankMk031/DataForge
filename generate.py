#!/usr/bin/env python3
"""
generate.py - Generate raw training samples using LLM

Usage:
    python generate.py --output raw_data.jsonl --samples 100 --domain speed_distance_time
    python generate.py --output raw_data.jsonl --samples 50 --domain all
"""

import json
import argparse
from typing import Optional
import time
from pathlib import Path
import random

from local_llm import call_local_llm

# ============================================================================
# CONFIGURATION
# ============================================================================

TASK_SPEC = {
    "name": "multi_step_algebra",
    "min_steps": 2,
    "max_steps": 5,
    "domains": [
        "speed_distance_time",
        "work_rate", 
        "mixture_problems",
        "profit_loss",
        "age_problems"
    ]
}

# Multiple prompt variants for diversity
PROMPT_TEMPLATES = {
    "detailed": """Generate a {difficulty} algebra word problem about {domain}.

Requirements:
- Exactly {num_steps} steps to solve
- Clear step-by-step reasoning
- Numerical answer with units
{domain_notes}

Output as JSON:
{{
  "question": "the word problem text",
  "reasoning": ["Step 1: ...", "Step 2: ...", ...],
  "answer": "final numerical answer with units",
  "difficulty": "{difficulty}",
  "domain": "{domain}",
  "num_steps": {num_steps}
}}""",

    "conversational": """You're creating a math practice problem.

Task: {difficulty} difficulty, {domain}, {num_steps} solution steps

Write it like you're teaching a student. Show your work clearly.
{domain_notes}

JSON format:
{{
  "question": "...",
  "reasoning": ["First, ...", "Then, ...", ...],
  "answer": "...",
  "difficulty": "{difficulty}",
  "domain": "{domain}",
  "num_steps": {num_steps}
}}""",

    "minimal": """Domain: {domain}
Difficulty: {difficulty}  
Steps: {num_steps}
{domain_notes}

Create problem with solution.

{{
  "question": "...",
  "reasoning": [...],
  "answer": "...",
  "difficulty": "{difficulty}",
  "domain": "{domain}",
  "num_steps": {num_steps}
}}"""
}

# ============================================================================
# LLM INTEGRATION (EDIT THIS SECTION)
# ============================================================================

def call_llm(prompt: str) -> str:
    """
    Generate a response using the local LLM helper (Ollama llama2 by default).
    """

    return call_local_llm(prompt)

# ============================================================================
# GENERATION LOGIC
# ============================================================================

def parse_llm_response(response: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response."""
    try:
        # Remove markdown code blocks
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        parsed = json.loads(response.strip())
        
        # Validate structure
        required_fields = ["question", "reasoning", "answer", "difficulty", "domain", "num_steps"]
        if all(field in parsed for field in required_fields):
            return parsed
        
        return None
    except:
        return None

def generate_sample(
    domain: str,
    difficulty: str,
    num_steps: int,
    variant: str,
    domain_notes: str,
) -> Optional[dict]:
    """Generate a single sample using specified prompt variant."""
    
    # Get prompt template
    template = PROMPT_TEMPLATES[variant]
    prompt = template.format(
        domain=domain,
        difficulty=difficulty,
        num_steps=num_steps,
        domain_notes=domain_notes,
    )
    
    # Try up to 3 times
    for attempt in range(3):
        try:
            response = call_llm(prompt)
            parsed = parse_llm_response(response)
            
            if parsed:
                # Add metadata
                parsed["prompt_variant"] = variant
                parsed["generation_timestamp"] = time.time()
                return parsed
            
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(1)
    
    return None

def generate_batch(
    domain: str,
    samples_per_config: int = 10,
    difficulties: list = None,
    step_range: tuple = (2, 5),
    domain_notes: str = "",
) -> list[dict]:
    """Generate a batch of samples for a domain."""
    
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]
    
    all_samples = []
    prompt_variants = list(PROMPT_TEMPLATES.keys())
    
    print(f"\n📊 Generating for domain: {domain}")
    
    configs = [(difficulty, num_steps) for difficulty in difficulties
               for num_steps in range(step_range[0], step_range[1] + 1)]
    random.shuffle(configs)

    for difficulty, num_steps in configs:
            
            config_samples = 0
            
            # Use all prompt variants for diversity
            for variant in prompt_variants:
                
                sample = generate_sample(domain, difficulty, num_steps, variant, domain_notes)
                
                if sample:
                    all_samples.append(sample)
                    config_samples += 1
                    
                    if config_samples >= samples_per_config:
                        break
            
            if config_samples > 0:
                print(f"  ✓ {difficulty}, {num_steps} steps: {config_samples} samples")
    
    return all_samples

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 samples for one domain
  python generate.py --output raw.jsonl --samples 100 --domain speed_distance_time
  
  # Generate 50 samples across all domains
  python generate.py --output raw.jsonl --samples 50 --domain all
  
  # Generate only hard problems
  python generate.py --output raw.jsonl --samples 30 --difficulty hard
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output JSONL file path (optional; default creates per-domain folders).",
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        help="Total number of samples to generate across all domains (overrides --samples-per-domain)"
    )

    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=200,
        help="Number of samples per domain (default: 200)",
    )

    parser.add_argument(
        "--run-dir",
        help="Base directory for per-domain outputs (default: runs/run_<timestamp>).",
    )
    
    parser.add_argument(
        "--domain", "-d",
        default="prompt",
        help=(
            f"Domain to generate. Options: {', '.join(TASK_SPEC['domains'])} or 'all'. "
            "Default: prompt for custom domains."
        ),
    )

    parser.add_argument(
        "--domains",
        help="Comma-separated custom domains (overrides --domain).",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for domains and their functionality notes.",
    )

    parser.add_argument(
        "--domain-notes",
        help="Comma-separated keywords to apply as functionality notes for all domains.",
    )
    
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty level (default: all)"
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        help="Minimum samples per domain; if any domain is below this number, retry generation.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum additional generation attempts if below --min-samples (default: 2)",
    )
    
    args = parser.parse_args()
    
    # Determine domains (support interactive/custom)
    domain_notes_map: dict[str, str] = {}
    if args.interactive or args.domain == "prompt":
        raw = input("Enter domains (comma-separated): ").strip()
        domains = [d.strip() for d in raw.split(",") if d.strip()]
        if not domains:
            print("Error: No domains provided.")
            return
        print("Enter functionality notes for each domain (short, comma-separated keywords ok).")
        for d in domains:
            notes = input(f"  {d}: ").strip()
            domain_notes_map[d] = notes
    elif args.domains:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        if not domains:
            print("Error: --domains was provided but empty.")
            return
    elif args.domain == "all":
        domains = TASK_SPEC["domains"]
    else:
        if args.domain not in TASK_SPEC["domains"]:
            print(f"Error: Unknown domain '{args.domain}'")
            print(f"Available: {', '.join(TASK_SPEC['domains'])}")
            return
        domains = [args.domain]
    
    # Determine difficulties
    if args.difficulty == "all":
        difficulties = ["easy", "medium", "hard"]
    else:
        difficulties = [args.difficulty]
    
    # Calculate samples per domain
    if args.samples is not None:
        samples_per_domain = args.samples // len(domains)
    else:
        samples_per_domain = args.samples_per_domain
    
    print("="*70)
    print("DATA GENERATION")
    print("="*70)
    if args.samples is not None:
        print(f"Target samples (total): {args.samples}")
    else:
        print(f"Target samples (per domain): {samples_per_domain}")
    print(f"Domains: {', '.join(domains)}")
    print(f"Difficulties: {', '.join(difficulties)}")
    print(f"Samples per domain: ~{samples_per_domain}")
    
    # Build domain notes
    domain_notes_by_domain: dict[str, str] = {}
    for d in domains:
        notes = domain_notes_map.get(d, "").strip()
        if not notes and args.domain_notes:
            notes = args.domain_notes.strip()
        if notes:
            domain_notes_by_domain[d] = f"\nDomain-specific focus:\n- {notes}\n"
        else:
            domain_notes_by_domain[d] = ""

    # Generate (with optional retries)
    all_samples = []
    target_min = args.min_samples
    retries_left = args.max_retries
    
    # Compute step_count from TASK_SPEC
    step_count = TASK_SPEC["max_steps"] - TASK_SPEC["min_steps"] + 1
    
    def _run_generation() -> list[dict]:
        generated = []
        for domain in domains:
            batch = generate_batch(
                domain=domain,
                samples_per_config=max(1, samples_per_domain // (len(difficulties) * step_count)),
                difficulties=difficulties,
                step_range=(TASK_SPEC["min_steps"], TASK_SPEC["max_steps"]),
                domain_notes=domain_notes_by_domain.get(domain, ""),
            )
            generated.extend(batch)
        return generated

    all_samples = _run_generation()

    def _below_minimum(samples: list[dict]) -> list[tuple[str, int]]:
        if not target_min:
            return []
        counts = {d: 0 for d in domains}
        for s in samples:
            d = s.get("domain")
            if d in counts:
                counts[d] += 1
        return [(d, c) for d, c in counts.items() if c < target_min]

    missing = _below_minimum(all_samples)
    while target_min and missing and retries_left > 0:
        missing_str = ", ".join(f"{d}={c}" for d, c in missing)
        print(
            f"\n⚠️  Samples below minimum per domain ({target_min}): {missing_str}. "
            f"Retrying... ({retries_left} retries left)"
        )
        retries_left -= 1
        all_samples = _run_generation()
        missing = _below_minimum(all_samples)

    # Save domain notes (if any)
    if any(domain_notes_map.values()):
        output_path = Path(args.output)
        config_path = output_path.with_name(f"{output_path.stem}_domain_config.json")
        with open(config_path, "w") as f:
            json.dump({"domains": domain_notes_map}, f, indent=2)
        print(f"\n🧩 Saved domain config: {config_path}")
    
    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output:
        print(f"\n💾 Saving to {args.output}...")
        with open(args.output, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")
    else:
        run_root = Path(args.run_dir) if args.run_dir else (Path("runs") / f"run_{timestamp}")
        run_root.mkdir(parents=True, exist_ok=True)
        print(f"\n💾 Saving per-domain samples under: {run_root}")
        for domain in domains:
            safe_domain = domain.replace("/", "_").strip()
            domain_dir = run_root / f"{safe_domain} data samples"
            domain_dir.mkdir(parents=True, exist_ok=True)
            raw_path = domain_dir / f"raw_{safe_domain}.jsonl"
            with open(raw_path, "w") as f:
                for sample in all_samples:
                    if sample.get("domain") == domain:
                        f.write(json.dumps(sample) + "\n")
            if domain in domain_notes_map:
                config_path = domain_dir / "domain_config.json"
                with open(config_path, "w") as f:
                    json.dump({"domains": {domain: domain_notes_map[domain]}}, f, indent=2)
        if any(domain_notes_map.values()):
            config_path = run_root / "domain_config.json"
            with open(config_path, "w") as f:
                json.dump({"domains": domain_notes_map}, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Generated: {len(all_samples)} samples")
    print(f"Output: {args.output}")
    
    # Stats
    from collections import Counter
    difficulties_count = Counter(s['difficulty'] for s in all_samples)
    domains_count = Counter(s['domain'] for s in all_samples)
    
    print(f"\nBy difficulty:")
    for diff, count in sorted(difficulties_count.items()):
        print(f"  {diff:8s}: {count:4d}")
    
    print(f"\nBy domain:")
    for domain, count in sorted(domains_count.items()):
        print(f"  {domain:25s}: {count:4d}")

if __name__ == "__main__":
    main()
