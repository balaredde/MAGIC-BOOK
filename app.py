import streamlit as st
from PIL import Image
import torch
import ollama
import time
import json
import os
import requests
import io
import base64
from gtts import gTTS
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    AutoImageProcessor, AutoModelForImageClassification
)

# Set Stability API key
os.environ["STABILITY_API_KEY"] = "sk-0tEtXE6mjW5pt7rj3mxAqLnLmZPPlvWPy6rRhJVqk9nPchaD"
API_HOST = os.getenv('API_HOST', 'https://api.stability.ai')

# Session state management
if 'story' not in st.session_state:
    st.session_state.story = None
if 'model_predictions' not in st.session_state:
    st.session_state.model_predictions = []
if 'custom_elements' not in st.session_state:
    st.session_state.custom_elements = {
        'captions': [],
        'keywords': [],
        'extra_keywords': []
    }
if 'narration_active' not in st.session_state:
    st.session_state.narration_active = False
if 'highlight_word' not in st.session_state:
    st.session_state.highlight_word = -1
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'voice_settings' not in st.session_state:
    st.session_state.voice_settings = {
        'voice_type': 'Eng-Uk',
        'speed': 1.0,
        'tld_map': {
            'Eng-Aus': ('com.au', False),
            'Eng-Uk': ('co.uk', True),
            'Eng-IND': ('co.in', False),
            'Eng-SA': ('co.za', False),
        }
    }
if 'narration_start_time' not in st.session_state:
    st.session_state.narration_start_time = 0
if 'audio_duration' not in st.session_state:
    st.session_state.audio_duration = 0

@st.cache_resource
def load_models():
    return {
        'blip': (
            BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
            BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        ),
        'vit_gpt2': (
            ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning"),
            VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning"),
            AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        ),
        'vit': (
            AutoImageProcessor.from_pretrained("google/vit-base-patch16-224"),
            AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        )
    }

def voice_settings_panel():
    with st.expander("🔊 Advanced Voice Settings"):
        col1, col2 = st.columns(2)
        with col1:
            voice_type = st.selectbox(
                "Voice Style",
                options=list(st.session_state.voice_settings['tld_map'].keys()),
                index=list(st.session_state.voice_settings['tld_map'].keys()).index(
                    st.session_state.voice_settings['voice_type']
                )
            )
        with col2:
            speed = st.slider(
                "Speech Speed",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.voice_settings['speed'],
                step=0.1
            )
        st.session_state.voice_settings.update({
            'voice_type': voice_type,
            'speed': speed
        })

def generate_image(prompt):
    """Generate image using Stability.ai API"""
    engine_id = "stable-diffusion-xl-1024-v1-0"
    api_key = os.environ["STABILITY_API_KEY"]

    if not api_key:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{API_HOST}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [{"text": prompt[:250]}],
            "cfg_scale": 9,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 40,
            "style_preset": "fantasy-art"
        },
    )

    if response.status_code != 200:
        st.error(f"Image generation failed: {response.text}")
        return None

    data = response.json()
    image_data = data["artifacts"][0]["base64"]
    return Image.open(io.BytesIO(base64.b64decode(image_data)))

def split_into_paragraphs(story_text):
    """Split story text into meaningful paragraphs"""
    return [p.strip() for p in story_text.split('\n\n') if p.strip()]

def generate_illustrations(story_text):
    """Generate images for each paragraph in the story"""
    paragraphs = split_into_paragraphs(story_text)
    generated_images = []
    
    with st.status("🎨 Generating illustrations..."):
        for i, para in enumerate(paragraphs):
            try:
                if len(para) < 15:
                    continue
                img = generate_image(para + " children's book illustration, colorful, magical, fantasy")
                if img:
                    generated_images.append(img)
                    st.write(f"Generated image for paragraph {i+1}")
            except Exception as e:
                st.error(f"Error generating image for paragraph {i+1}: {str(e)}")
    
    return generated_images

def analyze_images(images):
    models = load_models()
    all_predictions = []
    
    with st.status("🔍 Analyzing images..."):
        for image in images:
            predictions = {
                "captions": {
                    "BLIP": generate_blip_caption(image, *models['blip']),
                    "ViT-GPT2": generate_vit_gpt2_caption(image, *models['vit_gpt2'])
                },
                "labels": get_classification_labels(image, *models['vit'])
            }
            all_predictions.append(predictions)
            
    return all_predictions

def generate_blip_caption(image, processor, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=50)
    return processor.decode(outputs[0], skip_special_tokens=True)

def generate_vit_gpt2_caption(image, processor, model, tokenizer):
    inputs = processor(images=image, return_tensors="pt")
    generated_ids = model.generate(inputs.pixel_values, max_length=50)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def get_classification_labels(image, processor, model, top_k=5):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    top_probs, top_indices = probs.topk(top_k, dim=1)
    return [(model.config.id2label[idx.item()], prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])]

def generate_story(contexts):
    try:
        all_descriptions = [ctx['captions']['BLIP'] for ctx in contexts]
        all_labels = [label[0] for ctx in contexts for label in ctx['labels']]
        combined_elements = list(set(all_labels + st.session_state.custom_elements['extra_keywords']))
        
        prompt = f"""Create a children's story using these elements from multiple images:
        
        Image Descriptions: {'; '.join(all_descriptions)}
        Key Elements: {', '.join(combined_elements)}
        
        Format strictly as:
        Title: [Creative Story Title]
        Story: [Engaging story content in 3-5 paragraphs separated by blank lines]
        Moral: [Clear moral lesson]"""

        response = ollama.generate(
            model='llama3.2',
            prompt=prompt,
            options={'temperature': 0.7, 'max_tokens': 600}
        )

        sections = {'title': '', 'paragraphs': [], 'moral': ''}
        current_section = None
        current_paragraph = []
        
        for line in response['response'].split('\n'):
            line = line.strip()
            if line.startswith('Title:'):
                sections['title'] = line[6:].strip()
            elif line.startswith('Story:'):
                current_section = 'story'
            elif line.startswith('Moral:'):
                if current_paragraph:
                    sections['paragraphs'].append(' '.join(current_paragraph))
                    current_paragraph = []
                sections['moral'] = line[6:].strip()
                current_section = None
            elif current_section == 'story':
                if line == '':
                    if current_paragraph:
                        sections['paragraphs'].append(' '.join(current_paragraph))
                        current_paragraph = []
                else:
                    current_paragraph.append(line)

        if current_paragraph:
            sections['paragraphs'].append(' '.join(current_paragraph))

        return {
            'title': sections['title'] or "Magical Adventure",
            'paragraphs': sections['paragraphs'] or ["A wonderful adventure unfolds..."],
            'moral': sections['moral'] or "Imagination is powerful"
        }
        
    except Exception as e:
        st.error(f"Story generation failed: {str(e)}")
        return {
            'title': "The Enchanted Journey",
            'paragraphs': ["In a world where pictures come alive, these images hold secret wonders waiting to be discovered."],
            'moral': "Every picture tells a story worth imagining"
        }

def narration_controls(story):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.download_button(
            "📥 Download Story",
            data=json.dumps(story, indent=2),
            file_name="story.json",
            mime="application/json"
        )
    with col2:
        if not st.session_state.narration_active:
            if st.button("🔊 Narrate Story"):
                start_narration(story)
        else:
            st.button("🔊 Narrating...", disabled=True)
    with col3:
        if st.button("⏹ Stop Narration"):
            stop_narration()

def start_narration(story):
    try:
        stop_narration()
        full_text = f"{story['title']}. {' '.join(story['paragraphs'])} Moral: {story['moral']}"
        st.session_state.words = full_text.split()
        st.session_state.highlight_word = -1
        
        tld, slow = st.session_state.voice_settings['tld_map'][
            st.session_state.voice_settings['voice_type']
        ]
        speed_factor = st.session_state.voice_settings['speed']
        
        # Generate audio bytes
        with io.BytesIO() as audio_bytes:
            tts = gTTS(
                text=full_text,
                lang='en',
                tld=tld,
                slow=slow or (speed_factor < 0.8)
            )
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            audio_base64 = base64.b64encode(audio_bytes.getvalue()).decode()
            st.session_state.audio_data = f"data:audio/mp3;base64,{audio_base64}"
        
        # Estimate duration based on word count (4 words per second)
        word_count = len(st.session_state.words)
        st.session_state.audio_duration = word_count / 4
        st.session_state.narration_start_time = time.time()
        st.session_state.narration_active = True

    except Exception as e:
        st.error(f"Narration error: {str(e)}")
        st.session_state.narration_active = False

def stop_narration():
    st.session_state.narration_active = False
    st.session_state.highlight_word = -1
    # Inject JavaScript to stop audio
    stop_audio_js = """
    <script>
        var audioElements = document.getElementsByTagName('audio');
        for (var i = 0; i < audioElements.length; i++) {
            audioElements[i].pause();
            audioElements[i].currentTime = 0;
        }
    </script>
    """
    st.components.v1.html(stop_audio_js, height=0)

def customization_panel():
    with st.expander("📝 Customize Story Elements", expanded=True):
        st.subheader("Image Descriptions")
        for idx in range(len(st.session_state.model_predictions)):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.custom_elements['captions'][idx]['BLIP'] = st.text_input(
                    f"BLIP Caption (Image {idx+1})",
                    value=st.session_state.model_predictions[idx]['captions']['BLIP'],
                    key=f"blip_{idx}"
                )
            with col2:
                st.session_state.custom_elements['captions'][idx]['ViT-GPT2'] = st.text_input(
                    f"ViT-GPT2 Caption (Image {idx+1})",
                    value=st.session_state.model_predictions[idx]['captions']['ViT-GPT2'],
                    key=f"vit_{idx}"
                )
        
        st.subheader("Story Elements")
        all_labels = list(set([label[0] for pred in st.session_state.model_predictions for label in pred['labels']]))
        
        for i, label in enumerate(all_labels):
            st.session_state.custom_elements['keywords'].append(
                st.text_input(
                    f"Element {i+1}",
                    value=label,
                    key=f"og_{i}"
                )
            )
        
        for i in range(len(st.session_state.custom_elements['extra_keywords'])):
            st.session_state.custom_elements['extra_keywords'][i] = st.text_input(
                f"Additional Element {i+1}",
                value=st.session_state.custom_elements['extra_keywords'][i],
                key=f"extra_{i}"
            )
        
        if st.button("➕ Add New Element"):
            st.session_state.custom_elements['extra_keywords'].append("")

def main():
    st.set_page_config(page_title="Magic Storyteller", page_icon="🧙")
    st.title("🧙♂ Multi-Image Story Generator with Voice Control")
    
    voice_settings_panel()
    
    uploaded_files = st.file_uploader(
        "Upload images", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        images = [Image.open(file).convert("RGB") for file in uploaded_files]
        cols = st.columns(min(3, len(images)))
        for idx, (col, image) in enumerate(zip(cols, images)):
            col.image(image, use_container_width=True, caption=f"Image {idx+1}")
        
        if st.button("✨ Analyze Images"):
            st.session_state.model_predictions = analyze_images(images)
            st.session_state.custom_elements['captions'] = [
                {'BLIP': pred['captions']['BLIP'], 'ViT-GPT2': pred['captions']['ViT-GPT2']}
                for pred in st.session_state.model_predictions
            ]
            st.session_state.custom_elements['keywords'] = list(set(
                [label[0] for pred in st.session_state.model_predictions for label in pred['labels']]
            ))

    if st.session_state.model_predictions:
        customization_panel()
        
        if st.button("🚀 Generate Story", type="primary"):
            contexts = [{
                'captions': st.session_state.custom_elements['captions'][idx],
                'labels': st.session_state.model_predictions[idx]['labels']
            } for idx in range(len(st.session_state.model_predictions))]
            
            contexts[0]['extra_keywords'] = [
                kw for kw in st.session_state.custom_elements['extra_keywords'] 
                if kw.strip()
            ]
            
            with st.status("✨ Crafting your story..."):
                st.session_state.story = generate_story(contexts)
                st.session_state.generated_images = []

    if st.session_state.get('story'):
        st.divider()
        st.subheader(st.session_state.story['title'])
        
        # Auto-play audio when narration starts
        if st.session_state.narration_active:
            audio_html = f"""
            <audio autoplay>
                <source src="{st.session_state.audio_data}" type="audio/mp3">
            </audio>
            """
            st.components.v1.html(audio_html, height=0)
            
            # Update word highlighting based on estimated progress
            elapsed_time = time.time() - st.session_state.narration_start_time
            progress = min(elapsed_time / st.session_state.audio_duration, 1.0)
            current_word = int(progress * len(st.session_state.words))
            st.session_state.highlight_word = min(current_word, len(st.session_state.words)-1)
            
            # Check if audio should be stopped
            if elapsed_time >= st.session_state.audio_duration:
                stop_narration()
        else:
            st.session_state.highlight_word = -1
        
        for idx, paragraph in enumerate(st.session_state.story['paragraphs']):
            col1, col2 = st.columns([3, 2])
            with col1:
                words = paragraph.split()
                highlighted_paragraph = []
                for i, word in enumerate(words):
                    if i == st.session_state.highlight_word:
                        highlighted_paragraph.append(f"<mark>{word}</mark>")
                    else:
                        highlighted_paragraph.append(word)
                highlighted_text = " ".join(highlighted_paragraph)
                
                st.markdown(f"""
                <div style="line-height: 1.8; font-size: 16px; margin: 20px 0;">
                    {highlighted_text}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.session_state.generated_images and idx < len(st.session_state.generated_images):
                    st.image(
                        st.session_state.generated_images[idx],
                        caption=f"Paragraph {idx+1} Illustration",
                        use_container_width=True
                    )
                else:
                    st.write("Illustration will appear here after generation")
        
        st.markdown(f"🌟 Moral Lesson:** {st.session_state.story['moral']}")
        
        if st.button("🖼 Generate Story Illustrations"):
            st.session_state.generated_images = generate_illustrations(
                "\n\n".join(st.session_state.story['paragraphs'])
            )
            st.rerun()
        
        narration_controls(st.session_state.story)

if _name_ == "_main_":
    main()
