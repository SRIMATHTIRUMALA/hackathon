from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from groq import Groq
from flask import request, jsonify
import csv
from io import TextIOWrapper
import io
import random
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
# Load product dataset
with open('dmart_products.json', 'r', encoding='utf-8') as f:
    PRODUCTS = json.load(f)

# RAG Setup - Create embeddings for products
print("🔄 Building RAG index...")
product_texts = []
for p in PRODUCTS:
    text = f"{p['name']} {p['brand']} {p['category']} {p['subcategory']} {' '.join(p['health_tags'])} {' '.join(p['keywords'])}"
    product_texts.append(text)

vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
product_embeddings = vectorizer.fit_transform(product_texts)
print("✓ RAG index built successfully")

# Chat history storage (in production, use Redis/Database)
chat_sessions = {}

def search_products_rag(query, top_k=20):
    """RAG-based product search"""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, product_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            product = PRODUCTS[idx].copy()
            product['relevance_score'] = float(similarities[idx])
            results.append(product)
    
    return results

def filter_by_budget(products, budget):
    """Filter products within budget"""
    return [p for p in products if p['price'] <= budget]

def filter_by_health(products, health_goals):
    """Filter by health requirements"""
    if not health_goals:
        return products
    
    health_keywords = ['healthy', 'protein', 'low fat', 'organic', 'vitamin', 'fiber']
    filtered = []
    
    for product in products:
        health_tags_lower = [tag.lower() for tag in product['health_tags']]
        if any(keyword in ' '.join(health_tags_lower) for keyword in health_keywords):
            filtered.append(product)
    
    return filtered if filtered else products

def get_groq_response(user_message, context, chat_history):
    """Get AI response from Groq"""
    system_prompt = f"""You are a helpful DMart shopping assistant. Help users find products within their budget and meet their requirements.

AVAILABLE PRODUCTS CONTEXT:
{context}

Your task:
1. Understand user's budget and requirements
2. Recommend products from the context provided
3. Mention exact prices, rack locations, and brands
4. Suggest cheaper alternatives if budget is tight
5. Highlight health benefits when relevant
6. Be friendly and conversational

Always format product recommendations clearly with:
- Product name and brand
- Price (₹)
- Rack location
- Health benefits (if applicable)
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history
    for msg in chat_history[-6:]:  # Last 3 exchanges
        messages.append(msg)
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",

            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint with RAG"""
    data = request.json
    query = data.get('query', '')
    budget = data.get('budget', float('inf'))
    health_goal = data.get('health_goal', '')
    
    # RAG search
    products = search_products_rag(query, top_k=50)
    
    # Apply filters
    if budget < float('inf'):
        products = filter_by_budget(products, budget)
    
    if health_goal:
        products = filter_by_health(products, [health_goal])
    
    # Sort by relevance and price
    products.sort(key=lambda x: (x['relevance_score'], -x['price']), reverse=True)
    
    return jsonify({
        'success': True,
        'products': products[:20],
        'total_found': len(products)
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with Groq AI"""
    data = request.json
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    budget = data.get('budget')
    
    # Initialize session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    # Extract search intent from message
    search_query = user_message.lower()
    
    # RAG search for context
    products = search_products_rag(search_query, top_k=15)
    
    # Apply budget filter if specified
    if budget:
        products = filter_by_budget(products, float(budget))
    
    # Create context for AI
    context = "Here are relevant products from DMart:\n\n"
    for i, p in enumerate(products[:10], 1):
        context += f"{i}. {p['name']} by {p['brand']}\n"
        context += f"   Price: ₹{p['price']} (MRP: ₹{p['mrp']})"
        if p['discount_percent'] > 0:
            context += f" - {p['discount_percent']}% OFF"
        context += f"\n   Location: {p['rack_location']}\n"
        if p['health_tags']:
            context += f"   Health: {', '.join(p['health_tags'])}\n"
        context += f"   Stock: {'Available' if p['in_stock'] else 'Out of stock'}\n\n"
    
    # Get AI response
    chat_history = chat_sessions[session_id]
    ai_response = get_groq_response(user_message, context, chat_history)
    
    # Update chat history
    chat_sessions[session_id].append({"role": "user", "content": user_message})
    chat_sessions[session_id].append({"role": "assistant", "content": ai_response})
    
    return jsonify({
        'success': True,
        'response': ai_response,
        'products': products[:10]
    })

@app.route('/api/product/<product_id>')
def get_product(product_id):
    """Get single product details"""
    product = next((p for p in PRODUCTS if p['id'] == product_id), None)
    if product:
        return jsonify({'success': True, 'product': product})
    return jsonify({'success': False, 'message': 'Product not found'}), 404

@app.route('/api/categories')
def get_categories():
    """Get all categories"""
    categories = {}
    for product in PRODUCTS:
        cat = product['category']
        if cat not in categories:
            categories[cat] = []
        if product['subcategory'] not in categories[cat]:
            categories[cat].append(product['subcategory'])
    
    return jsonify({'success': True, 'categories': categories})

@app.route('/api/budget-recommendations', methods=['POST'])
def budget_recommendations():
    """Get product recommendations based on budget"""
    data = request.json
    budget = data.get('budget', 1000)
    preferences = data.get('preferences', [])
    
    # Search based on preferences
    all_products = []
    for pref in preferences:
        products = search_products_rag(pref, top_k=100)
        all_products.extend(products)
    
    # Remove duplicates
    seen = set()
    unique_products = []
    for p in all_products:
        if p['id'] not in seen:
            seen.add(p['id'])
            unique_products.append(p)
    
    # Filter by budget
    affordable = filter_by_budget(unique_products, budget)
    
    # Sort by value (discount + rating)
    affordable.sort(key=lambda x: (x['discount_percent'], x['rating']), reverse=True)
    
    # Calculate budget allocation
    total_cost = 0
    selected = []
    for product in affordable:
        if total_cost + product['price'] <= budget:
            selected.append(product)
            total_cost += product['price']
            if len(selected) >= 15:
                break
    
    return jsonify({
        'success': True,
        'selected_products': selected,
        'total_cost': round(total_cost, 2),
        'remaining_budget': round(budget - total_cost, 2)
    })

@app.route('/api/stats')
def get_stats():
    """Get dataset statistics"""
    total_products = len(PRODUCTS)
    categories = len(set(p['category'] for p in PRODUCTS))
    avg_price = sum(p['price'] for p in PRODUCTS) / total_products
    in_stock = sum(1 for p in PRODUCTS if p['in_stock'])
    
    return jsonify({
        'success': True,
        'stats': {
            'total_products': total_products,
            'categories': categories,
            'average_price': round(avg_price, 2),
            'in_stock': in_stock,
            'out_of_stock': total_products - in_stock
        }
    })
@app.route("/api/upload_products", methods=["POST"])
def upload_products():
    global PRODUCTS, product_embeddings, vectorizer

    file = request.files.get("file")
    if not file:
        return jsonify(success=False, error="No file uploaded"), 400

    try:
        # Read CSV content
        content = file.read().decode("utf-8")
        f = io.StringIO(content)
        reader = csv.DictReader(f)

        new_products = []
        count = 0
        for row in reader:
            # Convert CSV row dict into your product schema
            p = {
    "id": int(row["product_id"]),
    "category": row["category"],
    "subcategory": row["subcategory"],
    "name": row["name"],
    "brand": row["brand"],
    "size": row["size"],
    "price": float(row["price"]),
    "mrp": float(row["mrp"]),
    "discount_percent": float(row["discount_percent"]),
    "rack_location": row["rack_location"],
    "in_stock": row["in_stock"] in ("1","true","True"),
    "health_tags": [ ... ],   # from row["health_tags"]
    "keywords": [row["category"], row["brand"]],
}

            new_products.append(p)
            count += 1

        if count == 0:
            return jsonify(success=False, error="CSV has no rows"), 400

        # Replace global PRODUCTS
        PRODUCTS = new_products

        # Rebuild RAG index for new products
        product_texts = []
        for p in PRODUCTS:
            text = f"{p['name']} {p['brand']} {p['category']} {p.get('subcategory','')} {' '.join(p['health_tags'])} {' '.join(p['keywords'])}"
            product_texts.append(text)

        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        product_embeddings = vectorizer.fit_transform(product_texts)

        return jsonify(success=True, count=count)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
@app.route("/api/debug-search")
def debug_search():
    prods = search_products_rag("breakfast", top_k=5)
    return jsonify(count=len(prods), products=prods)
# Market Basket Analysis Setup
print("🔄 Building MBA rules...")
import random
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

MBA_RULES = []

def generate_sample_transactions(num_transactions=200):
    """Generate overlapping baskets so Apriori always finds patterns"""
    product_ids = [p["id"] for p in PRODUCTS]

    # If you somehow have very few products, bail out
    if len(product_ids) < 3:
        return []

    # Choose a small core set that appears very often (like rice, dal, oil)
    core_ids = product_ids[:5]        # first 5 products as "popular"
    other_ids = product_ids[5:]

    transactions = []
    for _ in range(num_transactions):
        basket = set()

        # Always pick 2–3 core items -> creates strong patterns
        core_size = random.randint(2, min(3, len(core_ids)))
        basket.update(random.sample(core_ids, core_size))

        # Sometimes add some others
        if other_ids:
            extra_size = random.randint(0, min(3, len(other_ids)))
            basket.update(random.sample(other_ids, extra_size))

        transactions.append(list(basket))

    return transactions

def build_mba_rules():
    global MBA_RULES
    transactions = generate_sample_transactions(200)
    if not transactions:
        print("⚠ MBA: Not enough products to build transactions")
        MBA_RULES = []
        return

    # One‑hot encode
    all_ids = sorted({pid for tx in transactions for pid in tx})
    rows = []
    for tx in transactions:
        row = {pid: (pid in tx) for pid in all_ids}
        rows.append(row)
    df = pd.DataFrame(rows)

    # Lower min_support because your universe is small
    frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
    if frequent_itemsets.empty:
        print("⚠ MBA: No frequent itemsets even with low support")
        MBA_RULES = []
        return

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
    if rules.empty:
        print("⚠ MBA: No rules after association_rules")
        MBA_RULES = []
        return

    MBA_RULES = rules.to_dict("records")
    print(f"✓ MBA: {len(MBA_RULES)} association rules created")

print("🔄 Building MBA rules...")
build_mba_rules()

@app.route("/api/mba-recommendations", methods=["POST"])
def mba_recommendations():
    data = request.json
    product_ids = data.get("product_ids", [])
    if not product_ids or not MBA_RULES:
        return jsonify(success=True, recommendations=[])

    base_ids = set(product_ids)
    seen = set(product_ids)
    recs = []

    for rule in MBA_RULES:
        ants = list(rule["antecedents"])
        cons = list(rule["consequents"])
        if all(a in base_ids for a in ants):
            for pid in cons:
                if pid not in seen:
                    prod = next((p for p in PRODUCTS if p["id"] == pid), None)
                    if prod and prod["in_stock"]:
                        item = prod.copy()
                        item["confidence"] = float(rule["confidence"])
                        recs.append(item)
                        seen.add(pid)
        if len(recs) >= 6:
            break

    recs.sort(key=lambda x: x["confidence"], reverse=True)
    return jsonify(success=True, recommendations=recs[:6])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)