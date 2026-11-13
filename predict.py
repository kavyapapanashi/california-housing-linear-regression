"""
California Housing Price Predictor - Interactive Inference Script
Task 1: Artificial Intelligence & Machine Learning

This script loads the trained LinearRegression model (model.pkl) and predicts
median house values based on user inputs or predefined regional presets.
"""

import os
import sys
import argparse
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')

PRESETS = {
    "1": {
        "name": "San Francisco Bay Area Coastal District",
        "MedInc": 8.5,        # $85,000 median income
        "HouseAge": 32.0,     # 32 years old
        "AveRooms": 6.4,      # 6.4 rooms
        "AveBedrms": 1.05,    # 1.05 bedrooms
        "Population": 1150.0, # 1,150 people
        "AveOccup": 2.5,      # 2.5 people/household
        "Latitude": 37.77,    # SF Latitude
        "Longitude": -122.42  # SF Longitude
    },
    "2": {
        "name": "Los Angeles Coastal Metro (Santa Monica)",
        "MedInc": 6.8,        # $68,000 median income
        "HouseAge": 28.0,
        "AveRooms": 5.8,
        "AveBedrms": 1.08,
        "Population": 1800.0,
        "AveOccup": 2.7,
        "Latitude": 34.02,
        "Longitude": -118.49
    },
    "3": {
        "name": "Inland Central Valley (Fresno / Bakersfield)",
        "MedInc": 2.8,        # $28,000 median income
        "HouseAge": 20.0,
        "AveRooms": 4.9,
        "AveBedrms": 1.06,
        "Population": 2200.0,
        "AveOccup": 3.4,
        "Latitude": 36.74,
        "Longitude": -119.78
    },
    "4": {
        "name": "Sacramento Suburban Community",
        "MedInc": 4.5,        # $45,000 median income
        "HouseAge": 22.0,
        "AveRooms": 5.5,
        "AveBedrms": 1.02,
        "Population": 1400.0,
        "AveOccup": 2.9,
        "Latitude": 38.58,
        "Longitude": -121.49
    }
}

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run build_all.py first.")
    return joblib.load(MODEL_PATH)

def format_prediction(features_dict, model):
    df_input = pd.DataFrame([features_dict])
    pred_val = model.predict(df_input)[0]
    dollar_val = pred_val * 100000

    print("\n" + "="*60)
    print("           PREDICTION RESULT SUMMARY")
    print("="*60)
    print("Input District Parameters:")
    print(f"  * Median Income (MedInc):   ${features_dict['MedInc']*10000:,.0f} ({features_dict['MedInc']:.2f})")
    print(f"  * Median House Age:         {features_dict['HouseAge']:.0f} years")
    print(f"  * Average Rooms:            {features_dict['AveRooms']:.2f} rooms/house")
    print(f"  * Average Bedrooms:         {features_dict['AveBedrms']:.2f} bedrooms/house")
    print(f"  * Block Population:         {features_dict['Population']:,.0f} people")
    print(f"  * Average Occupancy:        {features_dict['AveOccup']:.2f} people/household")
    print(f"  * Geo Coordinates:          {features_dict['Latitude']:.4f}° N, {features_dict['Longitude']:.4f}° W")
    print("-"*60)
    print(f"  ESTIMATED MEDIAN HOUSE VALUE:  ${dollar_val:,.2f}")
    print(f"  Normalized Model Unit:         {pred_val:.4f} ($100k)")
    print("="*60)
    
    if dollar_val > 500000:
        print("Note: The 1990 census capped luxury properties at $500,000.")
    elif dollar_val < 50000:
        print("Note: Estimate falls into the bottom 5th percentile of historical California housing.")
    print("\n")

def interactive_mode(model):
    print("\n" + "="*60)
    print("   CALIFORNIA HOUSING PRICE PREDICTOR - INTERACTIVE CLI")
    print("="*60)
    print("Choose an option:")
    print("  1 - San Francisco Bay Area Coastal District")
    print("  2 - Los Angeles Coastal Metro (Santa Monica)")
    print("  3 - Inland Central Valley (Fresno / Bakersfield)")
    print("  4 - Sacramento Suburban Community")
    print("  5 - Enter Custom District Features Manually")
    print("  q - Quit")
    print("="*60)

    choice = input("Enter selection [1-5 or q]: ").strip()
    if choice in ['q', 'Q', 'exit']:
        print("Exiting predictor. Goodbye!")
        return

    if choice in PRESETS:
        preset = PRESETS[choice]
        print(f"\n--> Selected Preset: {preset['name']}")
        feats = {k: v for k, v in preset.items() if k != 'name'}
        format_prediction(feats, model)
    elif choice == '5':
        print("\nEnter custom values (press Enter to accept default in brackets):")
        def ask_val(prompt, default):
            val = input(f"{prompt} [{default}]: ").strip()
            return float(val) if val else default

        try:
            med_inc = ask_val("Median income in $10,000s (e.g. 5.0 for $50k)", 5.0)
            house_age = ask_val("Median house age in years", 25.0)
            ave_rooms = ask_val("Average rooms per house", 5.5)
            ave_bedrms = ask_val("Average bedrooms per house", 1.05)
            population = ask_val("Total block population", 1200.0)
            ave_occup = ask_val("Average household occupancy", 2.8)
            latitude = ask_val("Latitude (California: ~32.5 to 42.0)", 36.5)
            longitude = ask_val("Longitude (California: ~-124.3 to -114.3)", -119.5)

            feats = {
                'MedInc': med_inc,
                'HouseAge': house_age,
                'AveRooms': ave_rooms,
                'AveBedrms': ave_bedrms,
                'Population': population,
                'AveOccup': ave_occup,
                'Latitude': latitude,
                'Longitude': longitude
            }
            format_prediction(feats, model)
        except ValueError:
            print("\n[Error] Invalid numerical input entered. Please try again.")
    else:
        print("\nInvalid choice. Please choose 1, 2, 3, 4, 5, or q.")

def main():
    model = load_trained_model()

    parser = argparse.ArgumentParser(description="Predict California Median House Value")
    parser.add_argument('--income', type=float, help="Median income in $10k (e.g., 4.5 for $45,000)")
    parser.add_argument('--age', type=float, help="Median house age (years)")
    parser.add_argument('--rooms', type=float, help="Average rooms per household")
    parser.add_argument('--bedrooms', type=float, help="Average bedrooms per household")
    parser.add_argument('--population', type=float, help="Total block population")
    parser.add_argument('--occupancy', type=float, help="Average household occupancy")
    parser.add_argument('--lat', type=float, help="Latitude")
    parser.add_argument('--lon', type=float, help="Longitude")
    parser.add_argument('--preset', type=str, choices=['1', '2', '3', '4'], help="Run a specific preset directly")

    args = parser.parse_args()

    if args.preset:
        preset = PRESETS[args.preset]
        print(f"Running Preset {args.preset}: {preset['name']}")
        feats = {k: v for k, v in preset.items() if k != 'name'}
        format_prediction(feats, model)
    elif args.income is not None:
        feats = {
            'MedInc': args.income,
            'HouseAge': args.age if args.age is not None else 28.0,
            'AveRooms': args.rooms if args.rooms is not None else 5.4,
            'AveBedrms': args.bedrooms if args.bedrooms is not None else 1.1,
            'Population': args.population if args.population is not None else 1400.0,
            'AveOccup': args.occupancy if args.occupancy is not None else 3.0,
            'Latitude': args.lat if args.lat is not None else 35.6,
            'Longitude': args.lon if args.lon is not None else -119.5
        }
        format_prediction(feats, model)
    else:
        interactive_mode(model)

if __name__ == '__main__':
    main()

