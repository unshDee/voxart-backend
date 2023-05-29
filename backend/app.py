import argparse
import base64
import os
from pathlib import Path
from io import BytesIO
import time

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from stable_diffusion_wrapper import StableDiffusionWrapper
from consts import DEFAULT_IMG_OUTPUT_DIR, MAX_FILE_NAME_LEN
from utils import parse_arg_boolean

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

app = Flask(__name__)
CORS(app)
print("--> Starting the image generation server. This might take up to two minutes.")

stable_diff_model = None

parser = argparse.ArgumentParser(description = "The API of VoxArt app to turn your prompts into visionary delights")
parser.add_argument("--port", type=int, default=8000, help = "backend port")
parser.add_argument("--save_to_disk", type = parse_arg_boolean, default = False, help = "Should save generated images to disk")
parser.add_argument("--img_format", type = str.lower, default = "JPEG", help = "Generated images format", choices=['jpeg', 'png'])
parser.add_argument("--output_dir", type = str, default = DEFAULT_IMG_OUTPUT_DIR, help = "Customer directory for generated images")
args = parser.parse_args()

def analyse_sentiment(input_text):
    # Define the input text
    # input_text = "plane crashing into a building and exploding"

    # Tokenize the input text
    tokens = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Perform sentiment analysis
    with torch.no_grad():
        logits = model(**tokens)[0]
        probabilities = torch.softmax(logits, dim=1).squeeze()
        predicted_class = torch.argmax(probabilities).item()

    # Print the sentiment label and confidence
    sentiment_labels = ["Negative", "Positive"]
    # print(f"Sentiment: {sentiment_labels[predicted_class]}")
    # print(f"Confidence: {probabilities[predicted_class] * 100:.2f}%")
    return [sentiment_labels[predicted_class], int(probabilities[predicted_class] * 100)]

@app.route("/generate", methods=["POST"])
@cross_origin()
def generate_images_api():
    json_data = request.get_json(force=True)
    text_prompt = json_data["text"]
    num_images = json_data["num_images"]
    
    sentiment = analyse_sentiment(text_prompt)
    
    warning = ""
    action = "No action taken."
    
    if (sentiment[0] == "Negative" and sentiment[1] >= 50.0):
        warning = "[HARMFUL]"
        action = "High chances of inappropriate image generation."
    
    elif (sentiment[0] == "Negative" and sentiment[1] < 50.0):
        warning = "[WARNING]"
        action = "Warning! Disturbing images might've been generated."
        
    generated_imgs = stable_diff_model.generate_images(text_prompt, num_images)

    returned_generated_images = []
    if args.save_to_disk:
        dir_name = os.path.join(args.output_dir,f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{warning + text_prompt}")[:MAX_FILE_NAME_LEN]
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    for idx, img in enumerate(generated_imgs):
        if args.save_to_disk: 
          img.save(os.path.join(dir_name, f'{idx}.{args.img_format}'), format=args.img_format)

        buffered = BytesIO()
        img.save(buffered, format=args.img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        returned_generated_images.append(img_str)

    print(f"Created {num_images} images from text prompt [{text_prompt}]")
    
    response = {
        'prompt': text_prompt,
        'generatedImgs': returned_generated_images,
        'generatedImgsFormat': args.img_format,
        'promptSentiment': sentiment[0],
        'sentimentConfidence': sentiment[1],
        'actionTaken': action
    }
    return jsonify(response)


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
    stable_diff_model = StableDiffusionWrapper()
    print("--> Image generation server is up and running!")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=False)
