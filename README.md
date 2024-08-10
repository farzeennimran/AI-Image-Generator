# AI Image Generator using Stable Diffusion

## Introduction

Welcome to the AI Image Generator project! This repository showcases how to generate high-quality images using the powerful capabilities of Stable Diffusion, a state-of-the-art AI model for text-to-image synthesis. With just a few lines of code, you can bring your creative ideas to life, generating images from textual prompts.

## What is Stable Diffusion?

Stable Diffusion is a deep learning model designed to generate images from text descriptions. Unlike many AI models, Stable Diffusion can run on local machines, making it highly accessible. It works by gradually transforming random noise into coherent images that match the given text prompt, leveraging a process called diffusion. This model is particularly known for its flexibility, high-quality output, and the ability to be fine-tuned for various applications.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.7+
- [Hugging Face Hub](https://huggingface.co/)
- diffusers
- transformers
- torch
- gradio

To generate an API token from huggingface go to the link, login to your account or sign up if you dont have one. Then, go to profile settings and click on 'Access Tokens'. Click on create new token, an API token will be generated. Save it and use that in your code. 

You can install the required libraries using pip:

```bash
!pip install diffusers transformers torch huggingface_hub gradio
```

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AI-image-generator.git
   cd AI-image-generator
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Login to Hugging Face:**
   Run the following code in a Jupyter notebook or script:
   ```python
   from huggingface_hub import notebook_login
   notebook_login()  # Enter your Hugging Face API key when prompted
   ```

4. **Set up the environment:**
   ```python
   import os
   os.environ['HUGGINGFACE_TOKEN'] = 'your_huggingface_token'
   ```

5. **Generate an image:**
   ```python
   from diffusers import StableDiffusionPipeline
   pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
   pipe = pipe.to("cuda")  # Use GPU if available

   prompt = "red full moon shown from earth"
   image = pipe(prompt).images[0]
   image.show()
   ```

6. **Use Gradio to create a web interface:**
   ```python
   import gradio as gr

   def generate_image(prompt):
       image = pipe(prompt).images[0]
       return image

   gr.Interface(fn=generate_image, inputs="text", outputs="image").launch()
   ```

## Explanation of the Code

- **StableDiffusionPipeline:** The core component that loads the pre-trained Stable Diffusion model from Hugging Face.
- **pipe(prompt).images[0]:** Generates an image based on the given prompt.
- **Gradio Interface:** A simple web interface to interact with the image generator using text inputs.

## Result

The project allows you to generate stunning images based on text prompts. For instance, the prompt `"red full moon shown from earth"` produces this image.

![ai-image](https://github.com/user-attachments/assets/c2b59752-3a76-4f72-a226-dd5d1fef35a6)

Gradio interface will look something like this

![gradio](https://github.com/user-attachments/assets/738e7c4c-ab51-4f9c-af95-adca01bdcf3c)

## Conclusion

This AI Image Generator demonstrates the power and versatility of Stable Diffusion in creating art from simple text prompts. Whether you're experimenting with creative ideas or building sophisticated applications, this project serves as a starting point for exploring the possibilities of AI-generated imagery.

Feel free to fork this repository, contribute, and share your unique creations!

## Learn More

For more information on AI image generation and the technology behind it, see my article: 
https://medium.com/@farzeeenimran/ai-image-generation-magic-myths-and-masterpieces-where-do-we-draw-the-line-d0955fd2c24a
