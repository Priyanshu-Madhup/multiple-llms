import streamlit as st
from groq import Groq
from openai import OpenAI
import base64
import os
import json
import http.client
from datetime import datetime
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# Streamlit page config
st.set_page_config(page_title="LLM Chat (Groq + OpenRouter + Serper)", page_icon="💬", layout="centered")

# Session state initialization
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "serper_api_key" not in st.session_state:
    st.session_state.serper_api_key = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "provider" not in st.session_state:
    st.session_state.provider = "Groq"
if "selected_model_name_groq" not in st.session_state:
    st.session_state.selected_model_name_groq = "Llama 3.3"
if "selected_model_name_openrouter" not in st.session_state:
    st.session_state.selected_model_name_openrouter = "Llama 3.3"
if "enable_serper" not in st.session_state:
    st.session_state.enable_serper = False
if "search_method" not in st.session_state:
    st.session_state.search_method = "POST"
if "image_generation_mode" not in st.session_state:
    st.session_state.image_generation_mode = False

# Model options per provider
model_options = {
    "Groq": {
        "Llama 3.3": "llama-3.3-70b-versatile",
        "DeepSeek R1": "deepseek-r1-distill-llama-70b",
        "Llama 4 Mavrick" : "meta-llama/llama-4-maverick-17b-128e-instruct",
        "Llama 4 Scout" : "meta-llama/llama-4-scout-17b-16e-instruct",
        "Mistral Saba" : "mistral-saba-24b"
    },
    "OpenRouter": {
        "DeepSeek R1": "deepseek/deepseek-r1:free",
        "DeepSeek V3": "deepseek/deepseek-chat-v3-0324:free",
        "Llama 4 Mavrick": "meta-llama/llama-4-maverick:free",
        "Llama 4 Scout": "meta-llama/llama-4-scout:free",
        "Llama 3.3": "meta-llama/llama-3.3-70b-instruct:free",
        #"Gemini 2.5 pro": "google/gemini-2.5-pro-exp-03-25:free",
        "Gemma 3": "google/gemma-3-27b-it:free"
    }
}

image_capable_models = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout:free",
    #"google/gemini-2.5-pro-exp-03-25:free",
    "google/gemma-3-27b-it:free",
    "mistral-saba-24b"
]

model_logos = {
    "DeepSeek R1": "deepseeklogo.png",
    "DeepSeek V3": "deepseeklogo.png",
    "Llama 4 Mavrick": "llamalogo.jpeg",
    "Llama 4 Scout": "llamalogo.jpeg",
    "Llama 3.3": "llamalogo.jpeg",
    "Gemini 2.5 pro": "gemini logo.png",
    "Gemma 3": "gemmalogo.jpeg",
    "Mistral Saba" : "mistrallogo.png"
}

# Function to search using Serper.dev with http.client (POST method)
def search_with_serper_post(query):
    if not st.session_state.serper_api_key:
        return {"error": "Serper API key is not set"}
    
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        
        payload = json.dumps({
            "q": query,
            "gl": "us",
            "hl": "en",
            "autocorrect": True
        })
        
        headers = {
            'X-API-KEY': st.session_state.serper_api_key,
            'Content-Type': 'application/json'
        }
        
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

# Function to search using Serper.dev with http.client (GET method)
def search_with_serper_get(query):
    if not st.session_state.serper_api_key:
        return {"error": "Serper API key is not set"}
    
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        
        # Format query for URL
        encoded_query = query.replace(' ', '+')
        endpoint = f"/search?q={encoded_query}&apiKey={st.session_state.serper_api_key}"
        
        conn.request("GET", endpoint)
        res = conn.getresponse()
        data = res.read()
        
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

# Function to format search results for the LLM
def format_search_results(results):
    if "error" in results:
        return f"Error searching: {results['error']}"
    
    formatted_results = f"Search results as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n\n"
    
    # Organic search results
    if "organic" in results:
        formatted_results += "Web Results:\n"
        for i, result in enumerate(results["organic"][:5], 1):
            formatted_results += f"{i}. {result.get('title', 'No title')}\n"
            formatted_results += f"   URL: {result.get('link', 'No link')}\n"
            if "snippet" in result:
                formatted_results += f"   Snippet: {result['snippet']}\n"
            formatted_results += "\n"
    
    # Knowledge Graph if available
    if "knowledgeGraph" in results:
        kg = results["knowledgeGraph"]
        formatted_results += "Knowledge Graph:\n"
        formatted_results += f"Title: {kg.get('title', 'N/A')}\n"
        if "description" in kg:
            formatted_results += f"Description: {kg['description']}\n"
        if "attributes" in kg:
            formatted_results += "Attributes:\n"
            for key, value in kg["attributes"].items():
                formatted_results += f"- {key}: {value}\n"
        formatted_results += "\n"
    
    # News results if available
    if "news" in results and results["news"]:
        formatted_results += "Recent News:\n"
        for i, news in enumerate(results["news"][:3], 1):
            formatted_results += f"{i}. {news.get('title', 'No title')}\n"
            formatted_results += f"   Source: {news.get('source', 'Unknown')}\n"
            formatted_results += f"   Published: {news.get('date', 'No date')}\n"
            formatted_results += f"   URL: {news.get('link', 'No link')}\n\n"
    
    # Answer boxes if available
    if "answerBox" in results:
        answer = results["answerBox"]
        formatted_results += "Featured Answer:\n"
        if "answer" in answer:
            formatted_results += f"Answer: {answer['answer']}\n"
        elif "snippet" in answer:
            formatted_results += f"Snippet: {answer['snippet']}\n"
        if "title" in answer:
            formatted_results += f"Source: {answer.get('title')}\n"
        if "link" in answer:
            formatted_results += f"URL: {answer.get('link')}\n"
        formatted_results += "\n"
    
    return formatted_results

# Function to generate images using Gemini
def generate_image_with_gemini(prompt):
    try:
        # Initialize client with the API key from session state
        client = genai.Client(api_key=st.session_state.gemini_api_key)
        
        # Generate image content
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        # Process response
        generated_image = None
        generation_text = ""
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                generation_text += part.text
            elif part.inline_data is not None:
                generated_image = BytesIO(part.inline_data.data)
        
        return generated_image, generation_text
    
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

# Function to get file download link for an image
def get_image_download_link(img_path, filename):
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}"><button style="margin-top: 8px; padding: 5px 10px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">📥 Download Image</button></a>'
    return href

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    provider = st.selectbox("Select Provider", ["Groq", "OpenRouter"], index=["Groq", "OpenRouter"].index(st.session_state.provider))
    st.session_state.provider = provider

    if provider == "Groq":
        model_names = list(model_options["Groq"].keys())
        selected_model_name = st.selectbox("Select Groq Model", options=model_names, index=model_names.index(st.session_state.selected_model_name_groq))
        st.session_state.selected_model_name_groq = selected_model_name
    else:
        model_names = list(model_options["OpenRouter"].keys())
        selected_model_name = st.selectbox("Select OpenRouter Model", options=model_names, index=model_names.index(st.session_state.selected_model_name_openrouter))
        st.session_state.selected_model_name_openrouter = selected_model_name

    # Unified selection
    st.session_state.selected_model_name = selected_model_name
    st.session_state.selected_model = model_options[provider][selected_model_name]

    st.info(f"Using {provider} / {selected_model_name}")
    st.caption(f"Model ID: {st.session_state.selected_model}")

    api_key_input = st.text_input(f"{provider} API Key", type="password", value=st.session_state.api_key)
    if st.button("Save API Key"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("API Key saved!")
            st.rerun()
        else:
            st.error("Please enter a valid API key.")

    # Web Search toggle first
    st.header("Web Search Integration")
    st.session_state.enable_serper = st.toggle("Enable Real-time Web Search", st.session_state.enable_serper)
    
    # Only show Serper settings if web search is enabled
    if st.session_state.enable_serper:
        # Serper.dev API key
        serper_api_key = st.text_input("Serper API Key", type="password", value=st.session_state.serper_api_key)
        if st.button("Save Serper API Key"):
            if serper_api_key:
                st.session_state.serper_api_key = serper_api_key
                st.success("Serper API Key saved!")
                st.rerun()
            else:
                st.error("Please enter a valid Serper API key.")
        
        # Search method selection - only shown if search is enabled
        st.session_state.search_method = st.radio(
            "API Request Method",
            options=["POST", "GET"],
            index=0 if st.session_state.search_method == "POST" else 1
        )
        
        st.caption("POST provides more control over search parameters. GET is simpler but more limited.")
        
        # Show Serper API key status
        if st.session_state.serper_api_key:
            st.success("Serper API Key is set")
        else:
            st.error("Serper API Key not set")

    # Image Generation section - Toggle first, then API key conditionally
    st.header("Image Generation")
    
    # Toggle for image generation mode first
    st.session_state.image_generation_mode = st.toggle("Generate Image Mode", st.session_state.image_generation_mode)
    
    # Only show API key input if image generation mode is enabled
    if st.session_state.image_generation_mode:
        st.info("Input will be treated as an image prompt when enabled")
        
        # Gemini API key input (only shown when toggle is on)
        gemini_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key)
        if st.button("Save Gemini API Key"):
            if gemini_api_key:
                st.session_state.gemini_api_key = gemini_api_key
                st.success("Gemini API Key saved!")
                st.rerun()
            else:
                st.error("Please enter a valid Gemini API key.")
        
        # Show Gemini API key status
        if st.session_state.gemini_api_key:
            st.success("Gemini API Key is set")
        else:
            st.warning("Gemini API Key not set - you'll need to set it to generate images")

    # Show main API key status
    if st.session_state.api_key:
        st.success("API Key is set")
    else:
        st.error("API Key not set")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Title with logo
try:
    image_path = model_logos.get(st.session_state.selected_model_name, None)
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            local_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        image_mime = "image/jpeg" if image_path.endswith(".jpeg") else "image/png"
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 10px;">
                <h1 style="margin: 0;">Multi LLM Chat</h1>
                <img src="data:{image_mime};base64,{local_img_base64}" height="100" style="margin-left: 10px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.title("Multi LLM Chat")
except Exception as e:
    st.error(f"Error loading logo: {str(e)}")
    st.title("Multi LLM Chat")

features_text = "Powered by " + st.session_state.selected_model_name + " via " + st.session_state.provider
if st.session_state.enable_serper:
    features_text += f" with Real-time Web Search ({st.session_state.search_method})"
if st.session_state.image_generation_mode:
    features_text += f" and Gemini Image Generation"
st.write(features_text)

# Display chat description based on model capabilities
is_image_capable = st.session_state.selected_model in image_capable_models
if is_image_capable:
    st.caption("This model supports image understanding. You can add images to your messages.")
else:
    st.caption("This model is text-only. Image attachments will be ignored.")

# Image generation mode status
if st.session_state.image_generation_mode:
    st.caption("Image generation mode is ON - your input will be used to generate an image")
    if not st.session_state.gemini_api_key:
        st.warning("Please set your Gemini API Key in the sidebar to use image generation")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image"):
            # Display the image
            st.image(message["image"], width=300)
            
            # Add download button for generated images (only in assistant messages with images)
            if message["role"] == "assistant" and os.path.exists(message["image"]):
                # Extract filename from path
                img_filename = os.path.basename(message["image"])
                # Add download button
                st.markdown(get_image_download_link(message["image"], img_filename), unsafe_allow_html=True)
        
        st.markdown(message["content"])

# Conditional chat input based on model capabilities
if is_image_capable:
    # Image-capable model: Use chat_input with file upload capability
    prompt_input = st.chat_input(
        "Type your message and/or attach an image",
        accept_file=True,
        file_type=["jpg", "jpeg", "png"],
    )
else:
    # Text-only model: Use simple chat_input
    prompt_input = st.chat_input("Type your message")

# Process input
if prompt_input:
    image_data = None
    
    # For image-capable models with file input
    if is_image_capable and hasattr(prompt_input, "files") and prompt_input.files:
        with st.spinner("Processing image..."):
            uploaded_file = prompt_input.files[0]
            image_bytes = uploaded_file.read()
            
            # Create a BytesIO object for display
            image_for_display = BytesIO(image_bytes)
            
            # Create a base64 encoded string for API
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_mime = "image/png" if uploaded_file.type == "image/png" else "image/jpeg"
            image_data_url = f"data:{image_mime};base64,{image_base64}"
            
            # Store both for different purposes
            image_data = {
                "display": image_for_display,
                "data_url": image_data_url
            }
            
            # Default message if only image is uploaded
            message_text = prompt_input.text if prompt_input.text else "What is in this image?"
    else:
        # For text-only input
        message_text = prompt_input if isinstance(prompt_input, str) else prompt_input.text
    
    # Add to message history
    user_message = {"role": "user", "content": message_text}
    if image_data:
        user_message["image"] = image_data["display"]
    
    st.session_state.messages.append(user_message)
    
    # Display the user message
    with st.chat_message("user"):
        if image_data:
            st.image(image_data["display"], width=300)
        st.markdown(message_text)

    # Check if image generation mode is enabled
    if st.session_state.image_generation_mode:
        # Check if Gemini API key is set
        if not st.session_state.gemini_api_key:
            error_msg = "⚠️ Gemini API Key is not set. Please set your API key in the sidebar to generate images."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)
        else:
            # This is an image generation request based on the toggle
            with st.spinner("Crafting your image..."):
                # Use the entire message as the image prompt
                image_prompt = message_text
                
                # Generate the image
                generated_image, generation_text = generate_image_with_gemini(image_prompt)
                
                # Prepare assistant's response
                if generated_image:
                    # Save image to a temporary file with timestamp to avoid overwrites
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    prompt_slug = image_prompt.lower().replace(' ', '_')[:30]  # Create a short slug from the prompt
                    img_filename = f"gemini_{prompt_slug}_{timestamp}.png"
                    temp_img_path = img_filename
                    
                    try:
                        image = Image.open(generated_image)
                        image.save(temp_img_path)
                        
                        # Create response text
                        response = f"Here's your generated image based on: '{image_prompt}'"
                        if generation_text:
                            response += f"\n\n{generation_text}"
                        
                        # Add to chat
                        assistant_message = {"role": "assistant", "content": response, "image": temp_img_path}
                        st.session_state.messages.append(assistant_message)
                        
                        # Display in chat
                        with st.chat_message("assistant"):
                            st.image(temp_img_path, width=300)
                            # Add download button
                            st.markdown(get_image_download_link(temp_img_path, img_filename), unsafe_allow_html=True)
                            st.markdown(response)
                            
                    except Exception as e:
                        error_msg = f"Error processing the generated image: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        with st.chat_message("assistant"):
                            st.markdown(error_msg)
                else:
                    error_msg = generation_text or "Failed to generate image."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    with st.chat_message("assistant"):
                        st.markdown(error_msg)
    else:
        # Search with Serper if enabled
        search_results = None
        if st.session_state.enable_serper and st.session_state.serper_api_key and message_text:
            with st.spinner(f"Searching for real-time information using {st.session_state.search_method} method..."):
                if st.session_state.search_method == "POST":
                    search_results = search_with_serper_post(message_text)
                else:
                    search_results = search_with_serper_get(message_text)
                
                formatted_search = format_search_results(search_results)
                
                # Display search status
                if "error" in search_results:
                    st.error(f"Search error: {search_results['error']}")
                else:
                    st.success("Search completed successfully")
        
        try:
            # Set spinner text based on whether an image is being analyzed
            spinner_text = "Analyzing the image..." if image_data else "Thinking..."
            
            with st.spinner(spinner_text):
                if st.session_state.provider == "Groq":
                    client = Groq(api_key=st.session_state.api_key)
                else:
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=st.session_state.api_key,
                    )

                # Create system message with search and image generation instructions
                system_content = "You are a helpful assistant."
                if st.session_state.enable_serper:
                    system_content += " You have access to real-time web search results. When providing information based on search results, cite the source. If the search results don't contain relevant information for a query, rely on your knowledge but acknowledge its limitations and time constraints."
                if st.session_state.image_generation_mode:
                    system_content += " You can generate images when the user toggles on the image generation mode."
                
                system_msg = {"role": "system", "content": system_content}
                api_messages = [system_msg]

                # Add conversation history
                for msg in st.session_state.messages:
                    if msg.get("image") and is_image_capable:
                        # For messages with images, we need to construct a content array
                        if msg is user_message and image_data:
                            # For the current message, use the freshly prepared image data
                            content_array = [
                                {"type": "text", "text": msg["content"]},
                                {"type": "image_url", "image_url": {"url": image_data["data_url"]}}
                            ]
                            api_messages.append({"role": msg["role"], "content": content_array})
                        else:
                            # Skip old messages with images if we don't have the data anymore
                            # This is a limitation as we're not storing the base64 data in the session
                            api_messages.append({"role": msg["role"], "content": msg["content"]})
                    else:
                        # For text-only messages
                        api_messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Add search results if available
                if search_results and formatted_search and "error" not in search_results:
                    api_messages.append({
                        "role": "system", 
                        "content": f"Here are real-time search results for the query: '{message_text}'\n\n{formatted_search}\n\nPlease use this information to help answer the user's question, citing sources when appropriate."
                    })

                if st.session_state.provider == "Groq":
                    chat_completion = client.chat.completions.create(
                        messages=api_messages,
                        model=st.session_state.selected_model,
                    )
                else:
                    chat_completion = client.chat.completions.create(
                        model=st.session_state.selected_model,
                        messages=api_messages,
                        extra_headers={
                            "HTTP-Referer": "http://localhost:8501",
                            "X-Title": "Streamlit Chat App",
                        },
                    )

                response = chat_completion.choices[0].message.content.strip()
                assistant_message = {"role": "assistant", "content": response}
                st.session_state.messages.append(assistant_message)

                with st.chat_message("assistant"):
                    st.markdown(response)

        except Exception as e:
            st.error(f"API Error: {str(e)}")