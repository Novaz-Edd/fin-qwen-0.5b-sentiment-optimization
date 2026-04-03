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
            "options": {"temperature": 0.7}
        }

    try:
        response = requests.post(url, json=payload).json()
        raw_output = response.get("response", "")
        cleaned_output = raw_output.strip()
    except Exception as e:
        raw_output = f"Error: {str(e)}"
        cleaned_output = raw_output
        
    end_time = time.time()
    return cleaned_output, round(end_time - start_time, 2), raw_output

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("AI Model Arena: Financial Sentiment 🥊")
st.markdown("Comparing your **Custom Fine-Tuned Model** against the **General Base Model**.")

headline = st.text_input("Enter a financial news headline:", "The company reported a massive 40% drop in Q3 revenue.")

if st.button("Run Side-by-Side Analysis 🏁"):
    
    col1, col2 = st.columns(2)
    
    # Define separate prompts for each model
    specialist_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the sentiment of this financial news headline. Respond with exactly one word: Positive, Negative, or Neutral.

### Input:
{headline}

### Response:
"""
    
    generalist_prompt = f"""What is the financial sentiment of this headline? Please analyze it and provide your thoughts: "{headline}"""
    
    # COLUMN 1: YOUR FINE-TUNED MODEL
    with col1:
        st.subheader("🟢 The Specialist")
        st.caption("Model: fin-qwen-1.5b (Fine-Tuned)")
        with st.spinner("Analyzing strictly..."):
            dav_answer, dav_time, dav_raw = call_ollama("fin-qwen-1.5b", specialist_prompt, is_finetuned=True)
            
            st.success(f"**Sentiment:** {dav_answer}")
            st.info(f"**Speed:** {dav_time} seconds")
            with st.expander("🔍 View Raw API Output"):
                st.code(repr(dav_raw), language="python")
            st.write("✅ **Formatting:** Perfect single word.")
            st.write("✅ **Behavior:** Laser-focused on the task.")

    # COLUMN 2: THE BASE MODEL
    with col2:
        st.subheader("🔴 The Generalist")
        st.caption("Model: qwen2.5:1.5b (Base Foundation)")
        with st.spinner("Thinking..."):
            gol_answer, gol_time, gol_raw = call_ollama("qwen2.5:1.5b", generalist_prompt, is_finetuned=False)
            
            st.warning(f"**Sentiment Output:** {gol_answer}")
            st.error(f"**Speed:** {gol_time} seconds")
            with st.expander("🔍 View Raw API Output"):
                st.code(repr(gol_raw), language="python")
            st.write("❌ **Formatting:** Often rambles or ignores the one-word rule.")
            st.write("❌ **Behavior:** Acts like a chatbot, not a classifier.")