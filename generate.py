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
    Generate a response using the local LLM helper (OPT-6.7b by default).
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

def generate_sample(domain: str, difficulty: str, num_steps: int, variant: str) -> Optional[dict]:
    """Generate a single sample using specified prompt variant."""
    
    # Get prompt template
    template = PROMPT_TEMPLATES[variant]
    prompt = template.format(
        domain=domain,
        difficulty=difficulty,
        num_steps=num_steps
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
    step_range: tuple = (2, 5)
) -> list[dict]:
    """Generate a batch of samples for a domain."""
    
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]
    
    all_samples = []
    prompt_variants = list(PROMPT_TEMPLATES.keys())
    
    print(f"\n📊 Generating for domain: {domain}")
    
    for difficulty in difficulties:
        for num_steps in range(step_range[0], step_range[1] + 1):
            
            config_samples = 0
            
            # Use all prompt variants for diversity
            for variant in prompt_variants:
                
                sample = generate_sample(domain, difficulty, num_steps, variant)
                
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
        required=True,
        help="Output JSONL file path"
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="Total number of samples to generate (default: 100)"
    )
    
    parser.add_argument(
        "--domain", "-d",
        default="all",
        help=f"Domain to generate. Options: {', '.join(TASK_SPEC['domains'])} or 'all' (default: all)"
    )
    
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty level (default: all)"
    )
    
    args = parser.parse_args()
    
    # Determine domains
    if args.domain == "all":
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
    samples_per_domain = args.samples // len(domains)
    
    print("="*70)
    print("DATA GENERATION")
    print("="*70)
    print(f"Target samples: {args.samples}")
    print(f"Domains: {', '.join(domains)}")
    print(f"Difficulties: {', '.join(difficulties)}")
    print(f"Samples per domain: ~{samples_per_domain}")
    
    # Generate
    all_samples = []
    
    for domain in domains:
        batch = generate_batch(
            domain=domain,
            samples_per_config=max(1, samples_per_domain // (len(difficulties) * 4)),
            difficulties=difficulties,
            step_range=(TASK_SPEC["min_steps"], TASK_SPEC["max_steps"])
        )
        all_samples.extend(batch)
    
    # Save
    print(f"\n💾 Saving to {args.output}...")
    
    with open(args.output, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
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
