"""
Gradio UI for Fine-tuned Llama Model
Provides an interactive interface for conversing with the fine-tuned LLM
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-1B"
LORA_WEIGHTS = "llama-3-1b-finetuned-lora"  # Change this to your model repo
MAX_LENGTH = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50

# Global variables for model caching
model = None
tokenizer = None
device = None

def load_model():
    """Load the base model and LoRA weights."""
    global model, tokenizer, device
    
    if model is not None:
        return model, tokenizer
    
    print("Loading model and tokenizer...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Try to load LoRA weights
    try:
        print(f"Loading LoRA weights from {LORA_WEIGHTS}...")
        model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
        print("LoRA weights loaded successfully!")
    except Exception as e:
        print(f"Note: Could not load LoRA weights ({e}). Using base model instead.")
    
    # Move to device if CPU
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    return model, tokenizer

def generate_text(prompt, max_length=MAX_LENGTH, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K):
    """Generate text using the fine-tuned model."""
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded. Please try again."
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move to device
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    except Exception as e:
        return f"Error during generation: {str(e)}"

def chat_interface(message, history):
    """Chat interface for multi-turn conversations."""
    
    # Build conversation context
    context = ""
    for user_msg, assistant_msg in history:
        context += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    
    # Create prompt
    prompt = f"{context}User: {message}\nAssistant:"
    
    # Generate response
    response = generate_text(prompt, max_length=256)
    
    # Extract only the assistant's response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Fine-tuned Llama Model UI") as demo:
        gr.Markdown("# 🤖 Fine-tuned Llama-3 1B Model")
        gr.Markdown(
            """
            This is a fine-tuned Large Language Model built with LoRA (Low-Rank Adaptation).
            
            **Features:**
            - Parameter-efficient fine-tuning using LoRA
            - Fast inference on CPU
            - Interactive chatbot interface
            
            **Model Details:**
            - Base Model: Llama-3.1 1B
            - Fine-tuned on: FineTome Instruction Dataset
            - Training Method: LoRA with 4-bit quantization
            """
        )
        
        with gr.Tabs():
            # Tab 1: Text Generation
            with gr.Tab("Text Generation"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=4
                        )
                        
                        with gr.Row():
                            max_length_slider = gr.Slider(
                                minimum=50,
                                maximum=512,
                                value=256,
                                step=10,
                                label="Max Length"
                            )
                            temperature_slider = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature"
                            )
                        
                        with gr.Row():
                            top_p_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top P"
                            )
                            top_k_slider = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=50,
                                step=5,
                                label="Top K"
                            )
                        
                        generate_button = gr.Button("Generate", variant="primary")
                    
                    with gr.Column():
                        output = gr.Textbox(
                            label="Generated Output",
                            interactive=False,
                            lines=8
                        )
                
                # Connect generate button
                generate_button.click(
                    fn=generate_text,
                    inputs=[prompt_input, max_length_slider, temperature_slider, top_p_slider, top_k_slider],
                    outputs=output
                )
            
            # Tab 2: Chat Interface
            with gr.Tab("Chat"):
                chatbot = gr.ChatInterface(
                    chat_interface,
                    examples=[
                        "What is machine learning?",
                        "Explain quantum computing in simple terms",
                        "Write a Python function to calculate Fibonacci numbers"
                    ],
                    title="Chat with Fine-tuned Llama",
                    description="Have a conversation with the fine-tuned LLM",
                )
        
        gr.Markdown(
            """
            ---
            ### About This Lab
            This project demonstrates:
            1. **Parameter-Efficient Fine-Tuning (PEFT)** using LoRA
            2. **Model Deployment** with Gradio on Hugging Face Spaces
            3. **Instruction Following** capabilities of fine-tuned LLMs
            
            For more information, see the [GitHub Repository](https://github.com/yourusername/llm-finetuning)
            """
        )
    
    return demo

if __name__ == "__main__":
    # Load model before starting the interface
    print("Loading model on startup...")
    load_model()
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(share=True)
