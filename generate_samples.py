import torch
from PIL import Image
import os
import diffusers

def generate_image(prompt, pipeline, num_inference_steps=50):
    try:
        with torch.no_grad():
            image = pipeline(prompt, num_inference_steps=num_inference_steps).images[0]
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def save_images(images, output_dir="C:/Users/KIIT/artistic_lora_enhancement/samples"):
    os.makedirs(output_dir, exist_ok=True)
    for i, (prompt, image) in enumerate(images):
        image.save(f"{output_dir}/generated_image_{i + 1}.png")
        print(f"Generated and saved image for prompt: '{prompt}'")

if __name__ == "__main__":
    from diffusers import StableDiffusionPipeline
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusionPipeline.from_pretrained("path_to_your_fine_tuned_model").to(device)
    
    # List of prompts
    prompts = [
        "A beautiful cat enjoying Hacktoberfest by DigitalOcean",
        "A futuristic cityscape at sunset with neon lights",
        "A serene landscape with mountains and a river"
    ]
    
    # Generate and save images
    images = [(prompt, generate_image(prompt, model)) for prompt in prompts]
    save_images(images)
