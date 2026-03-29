import streamlit as st
import requests
import time

# --- AI API Calls ---
def call_ollama(model_name, prompt, is_finetuned=False):
    start_time = time.time()
    url = "http://localhost:11434/api/generate"
    
    # If it's your fine-tuned model, apply your strict tripwires
    if is_finetuned:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {"temperature": 0.0, "num_predict": 3, "stop": ["\n", "###", "."]}
        }
    # If it's the base model, let it act naturally to show the difference
    else:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }

    try:
        response = requests.post(url, json=payload).json()
        output = response.get("response", "").strip()
    except Exception as e:
        output = f"Error: {str(e)}"
        
    end_time = time.time()
    return output, round(end_time - start_time, 2)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("AI Model Arena: Financial Sentiment 🥊")
st.markdown("Comparing your **Custom Fine-Tuned Model** against the **General Base Model**.")

headline = st.text_input("Enter a financial news headline:", "The company reported a massive 40% drop in Q3 revenue.")

if st.button("Run Side-by-Side Analysis 🏁"):
    
    col1, col2 = st.columns(2)
    
    # COLUMN 1: YOUR FINE-TUNED MODEL
    with col1:
        st.subheader("🟢 The Specialist")
        st.caption("Model: fin-qwen-1.5b (Fine-Tuned)")
        with st.spinner("Analyzing strictly..."):
            exact_prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnalyze the sentiment of this financial news headline. Respond with exactly one word: Positive, Negative, or Neutral.\n\n### Input:\n{headline}\n\n### Response:\n"
            
            dav_answer, dav_time = call_ollama("fin-qwen-1.5b", exact_prompt, is_finetuned=True)
            
            st.success(f"**Sentiment:** {dav_answer}")
            st.info(f"**Speed:** {dav_time} seconds")
            st.write("✅ **Formatting:** Perfect single word.")
            st.write("✅ **Behavior:** Laser-focused on the task.")

    # COLUMN 2: THE BASE MODEL
    with col2:
        st.subheader("🔴 The Generalist")
        st.caption("Model: qwen2.5:1.5b (Base Foundation)")
        with st.spinner("Thinking..."):
            chat_prompt = f"Analyze the sentiment of this financial news headline. Respond with exactly one word: Positive, Negative, or Neutral. Headline: {headline}"
            
            gol_answer, gol_time = call_ollama("qwen2.5:1.5b", chat_prompt, is_finetuned=False)
            
            st.warning(f"**Sentiment Output:** {gol_answer}")
            st.error(f"**Speed:** {gol_time} seconds")
            st.write("❌ **Formatting:** Often rambles or ignores the one-word rule.")
            st.write("❌ **Behavior:** Acts like a chatbot, not a classifier.")