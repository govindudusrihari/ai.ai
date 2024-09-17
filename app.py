import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2Model.from_pretrained("gpt2")

# Load fine-tuned model weights (if available)
try:
    model.load_weights("my_fine_tuned_gpt2_model.h5")
    print("Fine-tuned model loaded successfully.")
except Exception as e:
    print("Error loading fine-tuned model:", e)

# Define app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def generate_text():
    if request.method == "POST":
        prompt = request.form["prompt"]
        encoded_prompt = tokenizer.encode(prompt, return_tensors="tf")

        generated_text = model.generate(input_ids=encoded_prompt, max_length=50, num_beams=4)
        generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

        return render_template("result.html", generated_text=generated_text)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
