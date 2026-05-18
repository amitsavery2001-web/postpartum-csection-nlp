import os
import random
import pandas as pd

def generate_case_plans(total_samples=3000):
    """
    Generates a structured, attribute-conditioned Case Plan matrix for 3,000 samples.
    Ensures balanced splits for Train/Val/Test A and realistic imbalance for Test B.
    """
    # 1. Define the structural split dimensions
    train_size = int(total_samples * 0.70)       # 2100
    val_size = int(total_samples * 0.15)         # 450
    test_a_size = int(total_samples * 0.075)     # 225 (Balanced)
    test_b_size = int(total_samples * 0.075)     # 225 (Realistic Imbalanced)
    
    # List to hold all generated rows
    case_plans = []
    
    # 2. List of core clinical warning channels
    clinical_categories = [
        "infection_warning", "wound_problem", "bleeding_warning", 
        "severe_pain_warning", "urinary_problem", "respiratory_warning", "leg_clot_warning"
    ]
    
    # 3. Define linguistic persona options for dynamic pre-sampling
    speakers = ["patient", "caregiver"]
    anxiety_levels = ["low", "medium", "high"]
    literacy_levels = ["low", "medium", "high"]
    understatement_levels = ["low", "high"]
    lengths = ["short", "medium", "long"]
    directness_options = ["explicit", "indirect"]
    certainty_options = ["clear", "ambiguous"]
    
    for i in range(total_samples):
        # Determine which split this sample belongs to
        if i < train_size:
            split = "train"
        elif i < train_size + val_size:
            split = "validation"
        elif i < train_size + val_size + test_a_size:
            split = "test_balanced_A"
        else:
            split = "test_imbalanced_B"
            
        # Initialize labels dictionary to all zeros
        labels = {cat: 0 for cat in clinical_categories}
        labels["routine_recovery"] = 0
        
        # 4. Apply Split Strategy (Enforce balancing vs realistic clinical imbalance)
        if split in ["train", "validation", "test_balanced_A"]:
            # Balanced Splits: 50% chance of being a pure routine case, 50% chance of having symptoms
            is_routine = random.random() < 0.5
            if is_routine:
                labels["routine_recovery"] = 1
            else:
                # Sample 1 or 2 overlapping warning signs (Marginal frequency balancing)
                num_warnings = random.choice([1, 2])
                selected_warnings = random.sample(clinical_categories, num_warnings)
                for warning in selected_warnings:
                    labels[warning] = 1
                    
        else:
            # Test B (Realistic Clinical Stress Test): 80% routine recovery, 20% warning signs
            is_routine = random.random() < 0.80
            if is_routine:
                labels["routine_recovery"] = 1
            else:
                num_warnings = random.choice([1, 2])
                selected_warnings = random.sample(clinical_categories, num_warnings)
                for warning in selected_warnings:
                    labels[warning] = 1
        
        # 5. Pre-sample independent linguistic variables to prevent prompt replication
        plan = {
            "case_id": f"CASE_{i:04d}",
            "split": split,
            "speaker": random.choice(speakers),
            "anxiety": random.choice(anxiety_levels),
            "medical_literacy": random.choice(literacy_levels),
            "understatement": random.choice(understatement_levels) if labels["routine_recovery"] == 0 else "low",
            "message_length": random.choice(lengths),
            "directness": random.choice(directness_options),
            "symptom_certainty": random.choice(certainty_options),
            "postpartum_day": random.randint(1, 42) # Window: Day 1 up to Week 6
        }
        
        # Merge clinical labels into the plan execution payload
        plan.update(labels)
        case_plans.append(plan)
        
    # Convert into a pandas DataFrame and save to output paths
    df = pd.DataFrame(case_plans)
    return df

if __name__ == "__main__":
    # Ensure local directory structure exists before logging target profiles
    os.makedirs("data/raw", exist_ok=True)
    
    print("🚀 Initializing Attribute Pre-Sampling Generation Loop...")
    matrix_df = generate_case_plans(total_samples=3000)
    
    # Save the execution blueprint
    output_path = "data/raw/case_plans_blueprint.csv"
    matrix_df.to_csv(output_path, index=False)
    print(f"✅ Successfully generated and saved 3,000 case profiles to: {output_path}")
    
    # Print intermediate status to verify our dual-test distribution rule
    print("\n📊 Generated Class Balance Breakdown across splits:")
    print(matrix_df.groupby(["split", "routine_recovery"]).size())