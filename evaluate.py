import requests
import time

# 1. Your Test Dataset (Headline, True Sentiment)
test_data = [
    ("The company reported a massive 40% drop in Q3 revenue.", "Negative"),
    ("Profits soared by 20% this quarter exceeding all expectations.", "Positive"),
    ("The CEO announced he will be stepping down next year.", "Neutral"),
    ("Inflation fears cause the stock market to plummet.", "Negative"),
    ("The startup secured $50 million in Series B funding.", "Positive")
]

def test_model(headline):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the sentiment of this financial news headline. Respond with exactly one word: Positive, Negative, or Neutral.

### Input:
{headline}

### Response:
"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "fin-qwen-1.5b",
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0.0,
            "num_predict": 3,
            "stop": ["\n", "###", "."]
        }
    }
    
    start_time = time.time()
    response = requests.post(url, json=payload).json()
    end_time = time.time()
    
    return response.get("response", "").strip(), (end_time - start_time)

# 2. Run the Evaluation Race
print("Running Evaluation... 🏁\n")
correct_predictions = 0
total_time = 0

for headline, true_sentiment in test_data:
    print(f"Headline: {headline}")
    ai_prediction, time_taken = test_model(headline)
    total_time += time_taken
    
    # Check if the AI got it right
    if ai_prediction.lower() == true_sentiment.lower():
        print(f"✅ AI Guessed: {ai_prediction} | True: {true_sentiment} ({time_taken:.2f}s)")
        correct_predictions += 1
    else:
        print(f"❌ AI Guessed: {ai_prediction} | True: {true_sentiment} ({time_taken:.2f}s)")
    print("-" * 40)

# 3. Print the Final Report Card
accuracy = (correct_predictions / len(test_data)) * 100
avg_time = total_time / len(test_data)

print("\n📊 --- FINAL REPORT CARD --- 📊")
print(f"Total Tested: {len(test_data)}")
print(f"Accuracy: {accuracy}%")
print(f"Average Inference Time: {avg_time:.2f} seconds per prompt")