from dotenv import load_dotenv
import os
import csv
import random
import lancedb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from lancedb.pydantic import LanceModel, vector
import os

# Load environment variables from .env file
load_dotenv()

# Get API key and base URL from environment variables
# Ensure you have set OPENAI_API_KEY and OPENAI_BASE_URL in your .env file
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)



# Initialize the SentenceTransformer model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class RealEstate(LanceModel):
    vector: vector(384)
    neighborhood: str
    price: float
    bedrooms: int
    bathrooms: int
    house_size: int
    description: str
    neighborhood_description: str

# Function to generate embeddings for a given text
def generate_embedding(list):
    embedding = embedding_model.encode(list)
    return embedding

# Generate a listing
def generate_listing(num_listings = 5):
    
    listings = []
    for i in range(num_listings):

        neighborhood = f"{random.choice(neighborhoods)}"
        bedrooms = random.randint(1, 5)
        bathrooms = random.randint(1, bedrooms)
        house_size = random.randint(800, 3500)
        price = house_size * random.randint(200, 600)  # Price per sqft varies

        description = generate_description(prompt_template_desc.format(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            house_size=f"{house_size:,}",
            price=price
        ))
        neighborhood_description = generate_n_description(prompt_template_n_desc.format(neighborhood=neighborhood))
        listing = [
            neighborhood,
            f"${price:,.0f}",
            str(bedrooms),
            str(bathrooms),
            f"{house_size:,} sqft",
            description,
            neighborhood_description
        ]
        listings.append(listing)
    return listings


# Function to generate a real estate description using OpenAI
def generate_description(prompt_template):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt_template,
        max_tokens=150
    )
    description = response.choices[0].text.strip()
    return description

# Function to generate a real estate neighborhood description using OpenAI
def generate_n_description(prompt_template):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt_template,
        max_tokens=150
    )
    n_description = response.choices[0].text.strip()
    return n_description

# Function to save listings to a CSV file
def save_listings_to_csv(file_path, listings):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Neighborhood", "Price","Bedrooms",
                "Bathrooms",
                "House Size",
                "Description",
                "Neighborhood Description"
            ])
        for listing in listings:
            writer.writerow(listing)

# Function to read listings from a CSV file
def read_listings_from_csv(file_path):
    listings = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            listings.append(row)
    return listings

# Function to get user input for dream property
def get_user_dream_property():
    print("Please provide details about your dream property:")

    # Display the list of neighborhoods with numbers
    print("\nAvailable Neighborhoods:")
    for i, neighborhood in enumerate(neighborhoods, start=1):
        print(f"{i}. {neighborhood}")

    # Let the user pick a neighborhood by number
    while True:
        try:
            neighborhood_choice = int(input("\nPick a neighborhood by number: ").strip())
            if 1 <= neighborhood_choice <= len(neighborhoods):
                neighborhood = neighborhoods[neighborhood_choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(neighborhoods)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    price = input("Price (e.g., $500,000): ").strip()
    bedrooms = int(input("Number of Bedrooms: ").strip())
    bathrooms = int(input("Number of Bathrooms: ").strip())
    house_size = input("House Size (e.g., 2000 sqft): ").strip()
    description = input("Description of the property: ").strip()
    neighborhood_description = input("Description of the neighborhood: ").strip()

    return {
        "Neighborhood": neighborhood,
        "Price": price,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "House Size": house_size,
        "Description": description,
        "Neighborhood Description": neighborhood_description
    }

#Generates new listings with updated descriptions and neighborhood descriptions based on the user's dream property sentiment.
def generate_new_listings_with_sentiment(similar_properties, user_dream_property):
    new_listings = []

    for _, row in similar_properties.iterrows():
        # Prepare the prompt for generating a new description
        prompt = (
            f"Using the following sentiment as inspiration:\n"
            f"User Property Description: {user_dream_property['Description']}\n"
            f"User Neighborhood Description: {user_dream_property['Neighborhood Description']}\n\n"
            f"Generate a new property description for a listing with these details:\n"
            f"Neighborhood: {row['neighborhood']}\n"
            f"Price: {row['price']}\n"
            f"Bedrooms: {row['bedrooms']}\n"
            f"Bathrooms: {row['bathrooms']}\n"
            f"House Size: {row['house_size']} sqft\n"
            "Do not cut off in the middle of a sentence. Finish the last sentence if you reach the limit."
        )

        # Generate new descriptions using the LLM
        new_description = generate_description(prompt)
        new_neighborhood_description = generate_n_description(
            f"Generate a neighborhood description inspired by:\n"
            f"{user_dream_property['Neighborhood Description']}\n"
            f"For the neighborhood: {row['neighborhood']}"
            "Do not cut off in the middle of a sentence. Finish the last sentence if you reach the limit."
        )

        # Create a new listing with updated descriptions
        new_listing = {
            "Neighborhood": row['neighborhood'],
            "Price": row['price'],
            "Bedrooms": row['bedrooms'],
            "Bathrooms": row['bathrooms'],
            "House Size": row['house_size'],
            "Description": new_description,
            "Neighborhood Description": new_neighborhood_description
        }

        new_listings.append(new_listing)

    return new_listings

# Example usage
if __name__ == "__main__":


    # Name of the neighborhoods
    neighborhoods = [
        "Green Oaks",
        "Willow Creek",
        "River Bluffs",
        "Maple Ridge",
        "Sunset Hills",
        "Valley View"
    ]

    # Prompt templates for generating descriptions
    prompt_template_desc = (
        "Generate a real estate description with the following details:\n"
        "Price: {price}\n"
        "Bedrooms: {bedrooms}\n"
        "Bathrooms: {bathrooms}\n"
        "Size: {house_size} square meters\n"
        "Do not cut off in the middle of a sentence. Finish the last sentence if you reach the limit."
    )

    # Prompt template for generating neighborhood descriptions
    prompt_template_n_desc = (
        "Generate a real estate neighborhood description with the following details:\n"
        "Neighborhood: {neighborhood}\n"
        "Do not cut off in the middle of a sentence. Finish the last sentence if you reach the limit."
    )

    # Check if the CSV file exists and is not empty
    csv_file_path = "listings.csv"
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        print("Loading listings from CSV file...")
        listings = read_listings_from_csv(csv_file_path)
    else:
        print("CSV file not found or empty. Generating new listings...")
        listings = generate_listing(10)
        save_listings_to_csv(csv_file_path, listings)
        print("Listings generated and saved to CSV file.")

    # Generate embeddings for the listings
    print(listings[0])
    emb = generate_embedding(listings)

    db = lancedb.connect("~/.real_estate_db")

    data = [RealEstate(
        vector=emb[i],
        neighborhood=listings[i][0],
        price=float(listings[i][1].replace("$", "").replace(",", "")),
        bedrooms=int(listings[i][2]),
        bathrooms=int(listings[i][3]),
        house_size=int(listings[i][4].replace(" sqft", "").replace(",", "")),
        description=listings[i][5],
        neighborhood_description=listings[i][6]
    ) for i in range(len(listings))]

    table = db.drop_table("real_estate", ignore_missing=True)
    table = db.create_table(
        name="real_estate",
        schema=RealEstate,
        data=data
    )

    # Add a feature flag for testing
    USE_HARDCODED_DREAM_PROPERTY = True  # Set to True for testing, False for interactive input
    
    if USE_HARDCODED_DREAM_PROPERTY:
        # Hardcoded user_dream_property for testing
        user_dream_property = {
            "Neighborhood": "Valley View",
            "Price": "$600,000",
            "Bedrooms": 2,
            "Bathrooms": 2,
            "House Size": "2500 sqft",
            "Description": "A modern home with a large kitchen and garden.",
            "Neighborhood Description": "A peaceful area with parks and schools nearby."
        }
    else:
        # Get user input for dream property
        user_dream_property = get_user_dream_property()

    user_dream_property_embedding = generate_embedding([
        user_dream_property["Neighborhood"],
        user_dream_property["Price"],
        str(user_dream_property["Bedrooms"]),
        str(user_dream_property["Bathrooms"]),
        user_dream_property["House Size"],
        user_dream_property["Description"],
        user_dream_property["Neighborhood Description"]
    ])[0]

    # Search for similar properties
    similar_properties = table.search(user_dream_property_embedding).limit(5).to_pandas()
    output_strings = similar_properties.apply(
        lambda row: f"{row['neighborhood']} at the price: {row['price']} and DESCRIPTION {row['description']}",
        axis=1
    ).tolist()
    #print("\nSimilar properties found:", output_strings)

    #Criteria: Use of LLM for generating personalized descriptions
    new_listings = generate_new_listings_with_sentiment(similar_properties, user_dream_property)
    print("\nNew listings generated with updated descriptions:")
    for listing in new_listings:
        print(f"Neighborhood: {listing['Neighborhood']}")
        print(f"Price: {listing['Price']}")
        print(f"Bedrooms: {listing['Bedrooms']}")
        print(f"Bathrooms: {listing['Bathrooms']}")
        print(f"House Size: {listing['House Size']}")
        print(f"Description: {listing['Description']}")
        print(f"Neighborhood Description: {listing['Neighborhood Description']}")
        print("-" * 50)



# Summary of the whole program: Print user_dream_property - user preferences, top similar_properties - top matching item, and top new_listings item - top matchjing item with personalized description
print("\nUser Dream Property:")
print(f"Neighborhood: {user_dream_property['Neighborhood']}")
print(f"Price: {user_dream_property['Price']}")
print(f"Bedrooms: {user_dream_property['Bedrooms']}")
print(f"Bathrooms: {user_dream_property['Bathrooms']}")
print(f"House Size: {user_dream_property['House Size']}")
print(f"Description: {user_dream_property['Description']}")
print(f"Neighborhood Description: {user_dream_property['Neighborhood Description']}")

if not similar_properties.empty:
    top_similar_property = similar_properties.iloc[0]
    print("\nTop Similar Property:")
    print(f"Neighborhood: {top_similar_property['neighborhood']}")
    print(f"Price: {top_similar_property['price']}")
    print(f"Bedrooms: {top_similar_property['bedrooms']}")
    print(f"Bathrooms: {top_similar_property['bathrooms']}")
    print(f"House Size: {top_similar_property['house_size']} sqft")
    print(f"Description: {top_similar_property['description']}")
    print(f"Neighborhood Description: {top_similar_property['neighborhood_description']}")

if new_listings:
    top_new_listing = new_listings[0]
    print("\nTop New Listing (with updated descriptions):")
    print(f"Neighborhood: {top_new_listing['Neighborhood']}")
    print(f"Price: {top_new_listing['Price']}")
    print(f"Bedrooms: {top_new_listing['Bedrooms']}")
    print(f"Bathrooms: {top_new_listing['Bathrooms']}")
    print(f"House Size: {top_new_listing['House Size']}")
    print(f"Description: {top_new_listing['Description']}")
    print(f"Neighborhood Description: {top_new_listing['Neighborhood Description']}")