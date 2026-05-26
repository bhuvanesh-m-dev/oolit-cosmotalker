from flask import Flask, jsonify, request
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from cosmotalker import get_info

app = Flask(__name__)

# Model Configuration
MODEL_NAME = "google/gemma-3-270m-it"
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        try:
            print(f"Loading {MODEL_NAME} ... This may take a while.")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32
            )
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            print(f"{MODEL_NAME} loaded successfully.")
        except Exception as e:
            print(f"Model loading failed: {e}")

@app.route('/q', methods=['GET'])
def oolit_query():
    query = request.args.get('q')
    
    if not query:
        return jsonify({"status": "error", "message": "Query parameter 'q' is required"}), 400

    load_model()
    
    if model is None or tokenizer is None:
        return jsonify({"status": "error", "message": "AI model is not loaded yet"}), 503

    try:
        prompt = f"You are Oolit, a friendly astronomy and space assistant.\n\nUser: {query}\n\nOolit:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response_text.split("Oolit:")[-1].strip()
        
        return jsonify({
            "status": "success",
            "query": query,
            "response": response_text,
            "model": MODEL_NAME
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": "Failed to generate response"}), 500


@app.route('/api/get', methods=['GET'])
def get_space_info():
    query = request.args.get('q')
    if not query:
        return jsonify({"status": "error", "message": "Query required"}), 400
    data = get_info(query)
    return jsonify({"status": "success", "query": query, "data": data})

if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
