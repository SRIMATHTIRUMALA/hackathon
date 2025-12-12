import json
import random
from datetime import datetime, timedelta

# Product categories with subcategories
CATEGORIES = {
    "Fresh Vegetables": ["Leafy Greens", "Root Vegetables", "Exotic Vegetables", "Daily Vegetables"],
    "Fresh Fruits": ["Seasonal Fruits", "Exotic Fruits", "Citrus Fruits", "Berries"],
    "Dairy Products": ["Milk", "Curd & Yogurt", "Paneer & Cheese", "Butter & Ghee"],
    "Cooking Oils": ["Refined Oils", "Cold Pressed Oils", "Ghee", "Specialty Oils"],
    "Rice & Grains": ["Basmati Rice", "Regular Rice", "Other Grains", "Pulses"],
    "Spices & Masalas": ["Whole Spices", "Ground Spices", "Blended Masalas", "Organic Spices"],
    "Snacks": ["Chips", "Namkeen", "Biscuits", "Health Snacks"],
    "Beverages": ["Juices", "Soft Drinks", "Tea & Coffee", "Health Drinks"],
    "Personal Care": ["Soaps & Bodywash", "Shampoo & Conditioner", "Oral Care", "Skincare"],
    "Household": ["Cleaning Supplies", "Detergents", "Dishwashing", "Air Fresheners"],
    "Frozen Foods": ["Frozen Vegetables", "Frozen Snacks", "Ice Cream", "Ready to Eat"],
    "Breakfast & Cereals": ["Oats", "Cornflakes", "Muesli", "Instant Mixes"],
    "Bakery": ["Bread", "Cakes", "Cookies", "Buns & Pav"],
    "Meat & Seafood": ["Chicken", "Mutton", "Fish", "Eggs"],
    "Health & Nutrition": ["Protein Supplements", "Vitamins", "Organic Products", "Diet Foods"]
}

# Brand names for each category
BRANDS = {
    "Fresh Vegetables": ["Fresh", "Farm Fresh", "Organic Valley", "Green Harvest"],
    "Fresh Fruits": ["Fresh Picks", "Tropicana Fresh", "Farm Direct", "Premium Fresh"],
    "Dairy Products": ["Amul", "Mother Dairy", "Nandini", "Britannia", "Nestlé"],
    "Cooking Oils": ["Fortune", "Saffola", "Sundrop", "Dhara", "Gemini"],
    "Rice & Grains": ["India Gate", "Daawat", "Kohinoor", "Fortune", "Aashirvaad"],
    "Spices & Masalas": ["MDH", "Everest", "Catch", "Aashirvaad", "Eastern"],
    "Snacks": ["Lays", "Kurkure", "Haldiram", "Bikano", "Britannia"],
    "Beverages": ["Coca-Cola", "Pepsi", "Real", "Tropicana", "Tata Tea", "Nescafe"],
    "Personal Care": ["Dove", "Lux", "Lifebuoy", "Dettol", "Colgate", "Pepsodent"],
    "Household": ["Vim", "Harpic", "Lizol", "Surf Excel", "Ariel", "Tide"],
    "Frozen Foods": ["McCain", "Venky's", "Sumeru", "Prasuma", "ITC"],
    "Breakfast & Cereals": ["Quaker", "Kellogg's", "Saffola", "Bagrry's"],
    "Bakery": ["Britannia", "Harvest Gold", "Modern", "Bonn"],
    "Meat & Seafood": ["Venky's", "Zorabian", "Freshtohome", "Licious"],
    "Health & Nutrition": ["MuscleBlaze", "Protinex", "Horlicks", "Bournvita", "Organic India"]
}

# Product names for different categories
PRODUCTS = {
    "Leafy Greens": ["Spinach", "Methi", "Coriander", "Mint", "Lettuce", "Kale"],
    "Root Vegetables": ["Potato", "Onion", "Carrot", "Beetroot", "Radish", "Sweet Potato"],
    "Daily Vegetables": ["Tomato", "Capsicum", "Cauliflower", "Cabbage", "Brinjal", "Lady Finger"],
    "Seasonal Fruits": ["Apple", "Banana", "Mango", "Grapes", "Papaya", "Watermelon"],
    "Milk": ["Toned Milk", "Full Cream Milk", "Double Toned Milk", "Skimmed Milk"],
    "Curd & Yogurt": ["Fresh Curd", "Greek Yogurt", "Flavored Yogurt", "Probiotic Curd"],
    "Basmati Rice": ["White Basmati", "Brown Basmati", "Aged Basmati", "Organic Basmati"],
    "Pulses": ["Toor Dal", "Moong Dal", "Masoor Dal", "Chana Dal", "Urad Dal"],
    "Whole Spices": ["Cumin Seeds", "Coriander Seeds", "Black Pepper", "Cardamom", "Cinnamon"],
    "Chips": ["Classic Salted", "Masala", "Tomato", "Cream & Onion"],
    "Juices": ["Orange Juice", "Apple Juice", "Mixed Fruit", "Mango Juice"],
    "Soaps & Bodywash": ["Bathing Soap", "Body Wash", "Handwash", "Moisturizing Soap"],
    "Cleaning Supplies": ["Floor Cleaner", "Glass Cleaner", "Bathroom Cleaner", "Multi-purpose Cleaner"]
}

# Health tags
HEALTH_TAGS = ["High Protein", "Low Fat", "Organic", "Gluten Free", "Sugar Free", 
               "Heart Healthy", "Rich in Fiber", "Vitamin Rich", "Low Calorie", 
               "Diabetic Friendly", "Vegan", "Natural"]

# Rack organization (Aisle-Rack-Shelf)
RACKS = {}
aisle_num = 1
for category in CATEGORIES.keys():
    rack_num = random.randint(1, 8)
    shelf_num = random.randint(1, 5)
    RACKS[category] = f"A{aisle_num:02d}-R{rack_num:02d}-S{shelf_num}"
    if random.random() > 0.6:
        aisle_num += 1

def generate_product_name(category, subcategory):
    """Generate realistic product name"""
    base_products = PRODUCTS.get(subcategory, [subcategory])
    base = random.choice(base_products)
    
    # Add variants
    variants = ["Premium", "Fresh", "Organic", "Special", "Select", "Royal", "Classic"]
    sizes = ["500g", "1kg", "2kg", "250ml", "500ml", "1L", "2L", "100g", "250g"]
    
    if random.random() > 0.7:
        return f"{random.choice(variants)} {base} {random.choice(sizes)}"
    return f"{base} {random.choice(sizes)}"

def generate_dataset(num_records=15000):
    """Generate comprehensive DMart product dataset"""
    products = []
    product_id = 1
    
    records_per_category = num_records // len(CATEGORIES)
    
    for category, subcategories in CATEGORIES.items():
        brands = BRANDS.get(category, ["Generic", "Store Brand"])
        
        for _ in range(records_per_category):
            subcategory = random.choice(subcategories)
            brand = random.choice(brands)
            product_name = generate_product_name(category, subcategory)
            
            # Price logic based on category
            if category in ["Fresh Vegetables", "Fresh Fruits"]:
                base_price = random.uniform(20, 150)
            elif category in ["Dairy Products", "Bakery"]:
                base_price = random.uniform(25, 200)
            elif category in ["Meat & Seafood"]:
                base_price = random.uniform(150, 800)
            elif category in ["Health & Nutrition"]:
                base_price = random.uniform(200, 2500)
            else:
                base_price = random.uniform(30, 500)
            
            # Discount
            discount = random.choice([0, 5, 10, 15, 20, 25]) if random.random() > 0.5 else 0
            final_price = round(base_price * (1 - discount/100), 2)
            
            # Stock
            stock = random.randint(0, 500)
            
            # Health tags
            num_tags = random.randint(0, 3)
            health_tags = random.sample(HEALTH_TAGS, num_tags) if num_tags > 0 else []
            
            # Nutrition (for relevant categories)
            nutrition = None
            if category in ["Fresh Vegetables", "Fresh Fruits", "Dairy Products", 
                          "Breakfast & Cereals", "Health & Nutrition", "Meat & Seafood"]:
                nutrition = {
                    "calories": random.randint(50, 500),
                    "protein": round(random.uniform(0, 30), 1),
                    "carbs": round(random.uniform(0, 80), 1),
                    "fat": round(random.uniform(0, 40), 1),
                    "fiber": round(random.uniform(0, 15), 1)
                }
            
            product = {
                "id": f"DMART{product_id:06d}",
                "name": product_name,
                "brand": brand,
                "category": category,
                "subcategory": subcategory,
                "price": final_price,
                "mrp": round(base_price, 2),
                "discount_percent": discount,
                "stock": stock,
                "rack_location": RACKS[category],
                "health_tags": health_tags,
                "nutrition": nutrition,
                "rating": round(random.uniform(3.5, 5.0), 1),
                "reviews": random.randint(10, 1000),
                "in_stock": stock > 0,
                "weight_volume": random.choice(["100g", "250g", "500g", "1kg", "2kg", "250ml", "500ml", "1L", "2L"]),
                "expiry_days": random.randint(30, 730) if category not in ["Household", "Personal Care"] else None,
                "description": f"High quality {product_name} from {brand}. Available at DMart stores.",
                "keywords": [category.lower(), subcategory.lower(), brand.lower()] + [tag.lower() for tag in health_tags]
            }
            
            products.append(product)
            product_id += 1
    
    # Add extra records to reach 15000
    remaining = num_records - len(products)
    if remaining > 0:
        products.extend(random.sample(products, min(remaining, len(products))))
    
    return products[:num_records]

# Generate dataset
print("Generating 15,000+ product dataset...")
dataset = generate_dataset(15000)

# Save to JSON
with open('dmart_products.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"✓ Generated {len(dataset)} products")
print(f"✓ Categories: {len(CATEGORIES)}")
print(f"✓ Sample product: {dataset[0]['name']} - ₹{dataset[0]['price']}")
print("✓ Dataset saved to dmart_products.json")

# Generate summary statistics
print("\n📊 Dataset Statistics:")
print(f"Total Products: {len(dataset)}")
print(f"Average Price: ₹{sum(p['price'] for p in dataset)/len(dataset):.2f}")
print(f"Products with Health Tags: {sum(1 for p in dataset if p['health_tags'])}")
print(f"In Stock Products: {sum(1 for p in dataset if p['in_stock'])}")