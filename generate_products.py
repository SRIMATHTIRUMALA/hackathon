import csv
import random

categories = [
    ("Rice", ["Regular Rice", "Basmati Rice", "Sona Masoori"]),
    ("Biscuits", ["Marie Gold", "Good Day", "Bourbon", "Milk Bikis"]),
    ("Dairy", ["Toned Milk", "Curd", "Paneer", "Cheese Slices"]),
    ("Snacks", ["Potato Chips", "Namkeen", "Mixture", "Khakhra"]),
    ("Oil", ["Sunflower Oil", "Groundnut Oil", "Mustard Oil"]),
    ("Pulses", ["Toor Dal", "Chana Dal", "Moong Dal"]),
    ("Beverages", ["Orange Juice", "Mango Drink", "Cola", "Soda"]),
    ("Bread", ["Whole Wheat Bread", "White Bread", "Brown Bread"])
]

brands = ["DMart", "Aashirvaad", "Fortune", "India Gate", "Britannia", "Amul", "Parle", "Lays", "Modern"]
sizes = ["100 g", "200 g", "250 g", "500 g", "1 kg", "200 ml", "500 ml", "1 L", "2 kg"]
health_tags_list = [
    "staple", "high fibre", "low sugar", "protein",
    "low fat", "diabetic friendly", "kids", "snack"
]

def random_rack(i):
    aisle = f"A{1 + (i % 10):02d}"
    row = f"R{1 + (i % 5):02d}"
    shelf = f"S{1 + (i % 6):02d}"
    return f"{aisle}-{row}-{shelf}"

with open("dmart_products.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "product_id","category","name","brand","size",
        "price","mrp","discount_percent","rack_location","in_stock","health_tags"
    ])

    product_id = 1
    for i in range(5000):
        category, base_names = random.choice(categories)
        base_name = random.choice(base_names)
        brand = random.choice(brands)
        size = random.choice(sizes)

        name = f"{base_name} {size} {brand}"

        mrp = round(random.uniform(20, 800), 2)
        discount_percent = random.choice([0, 5, 10, 15, 20])
        price = round(mrp * (100 - discount_percent) / 100, 2)

        rack_location = random_rack(i)
        in_stock = 1 if random.random() > 0.1 else 0

        tags = ";".join(random.sample(health_tags_list, k=random.randint(1, 3)))

        writer.writerow([
            product_id, category, name, brand, size,
            price, mrp, discount_percent, rack_location, in_stock, tags
        ])
        product_id += 1

print("dmart_products.csv generated with 5000 rows")
