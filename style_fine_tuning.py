import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import StableDiffusionPipeline, get_scheduler
from tqdm import tqdm

# Use the GPU if it's available; otherwise, fall back to the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the base Stable Diffusion model
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v-1-4").to(device)

# Setting up image transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize images to 512x512
    transforms.ToTensor(),          # Convert images to tensors
])

# Function to load the dataset from a specific directory
def load_dataset(data_dir):
    dataset = ImageFolder(root=data_dir, transform=transform)  # Load images from folders
    return DataLoader(dataset, batch_size=4, shuffle=True)      # Create a data loader

# Function to fine-tune the model
def fine_tune_model(data_loader, epochs=1, learning_rate=1e-5):
    # Setting up the optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(data_loader) * epochs)
    model.train()  # Switching the model to training mode

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in tqdm(data_loader):
            images = batch[0].to(device)  # Moving images to the appropriate device
            optimizer.zero_grad()  # Clearing the gradients for the optimizer
            # Forward pass through the model
            outputs = model(images)
            loss = compute_loss(outputs)  # Calculating the loss (I'll define this function below)
            # Backward pass to calculate gradients
            loss.backward()
            optimizer.step()  # Updating the model weights
            scheduler.step()  # Updating the learning rate
            print(f"Loss: {loss.item()}")  # Print the current loss

    # After training, save the LoRA weights
    torch.save(model.lora_weights.state_dict(), "lora_weights.pt")
    print("LoRA weights saved as lora_weights.pt")

# Simple loss function (replace this with your actual loss calculation)
def compute_loss(outputs):
    # This is a dummy loss function just for illustration
    return torch.mean((outputs - 0.5) ** 2)  # Adjust this to fit your needs

if __name__ == "__main__":
    # Load dataset from a specific directory
    data_dir = "C:/Users/KIIT/artistic_lora_enhancement/dataset"  
    data_loader = load_dataset(data_dir)
    # Start the fine-tuning process
    fine_tune_model(data_loader, epochs=5, learning_rate=1e-5)
