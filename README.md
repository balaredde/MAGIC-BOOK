# 🧙 Multi-Image Story generator 


**Multi-Image Story Generator with Voice Control**

# Problem Statement Addressed
This project aims to create a platform where images uploaded by users are analyzed and used to generate a  children's story.  system allows users to narrate the story with customizable voice settings and generates images for each story paragraph.



# Objective and Goals of the Project
- Generate Creative Stories
- story narration
- image generation
- user friendly


# Key Features and Functionalities
- can narrate the story
- can download the story
- we can add elements in a dynamic way

# Target Audience
- Parents: Who want to create custom bedtime stories for their children.
- Teachers: can be used in story telling
- Developers: exploring the integration of AI models 
- kids: exploring new things , interactive


#Technology Stack

- Frontend: Streamlit (Python-based framework for web apps)
- Image Captioning: BLIP (Salesforce), ViT-GPT2 (Transformer-based models)
- Image Classification: Google ViT (Vision Transformer)
- Text-to-Speech: Google Text-to-Speech (gTTS)
- Image Generation: Stability.ai API (Stable Diffusion model)
- Backend:mango db
- Cloud Services: Stability.ai API for image generation

# Versions:
- Streamlit: 1.8.0
- Torch: 1.10.0
- Transformers: 4.12.0
- gTTS: 2.2.3



# System Requirements

1) Hardware Requirements:
- Minimum: 2 GB RAM, 2 CPU cores
- Recommended: 8 GB RAM, 4+ CPU cores

2) Software Requirements:
- Python 3.8+
- Dependencies (see `requirements.txt` for a complete list):
  - Streamlit
  - Torch
  - Transformers
  - gTTS
  - Requests
  - Pillow
  - Ollama

# Installation and Setup Instructions

 1. Clone the Repository
```bash
git clone https://github.com/your-repo/multi-image-story-generator.git
cd multi-image-story-generator


# Features Explanation

- user can upload multiple images
- user can have a look of image description
- able to download the story
- user can narrate the story and some advanced features are also present such as speed adjustment , voice changer
- user can save the history through a user id and password


# Usage Inustructions

- Upload images.
- View the generated story.
- Customize the story.
- Listen to the narration.
- Download the story

# Code Structure
|-app.py


#Testing 

tested the model using the images from different fields
compared the keywords with the images to see the accuracy levels


#Challenges 

1) integration with API
2) creating database


# Future enchancements

-More Voice Styles
-more languages
-Real-Time Collaboration
-Mobile Compatibility
-creativity levels


# Credits and References
Streamlit: https://streamlit.io
Stability.ai API: https://stability.ai
gTTS: https://pypi.org/project/gTTS/
BLIP and ViT Models: https://huggingface.co
