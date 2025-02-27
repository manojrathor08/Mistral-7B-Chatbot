import gradio as gr
from llm_chat import generate_response  # Import LLM function
from huggingface_hub import InferenceClient
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

MAX_HISTORY = 2  # Keep last 5 exchanges
SUMMARIZE_AFTER = 3  # Summarize after every 6 exchanges

def summarize_conversation(history):
    """
    Summarizes the conversation to maintain context efficiently.
    """
    summary_prompt = "Summarize this conversation briefly:\n\n"
    
    for user_msg, bot_reply in history:
        summary_prompt += f"User: {user_msg}\nAssistant: {bot_reply}\n"

    summary_prompt += "\nSummary:"
    
    # Generate the summary using the same model
    summary = ""
    for msg in client.chat_completion(
        [{"role": "user", "content": summary_prompt}], 
        max_tokens=100,  # Limit summary length
        temperature=0.3,  # Lower temp for consistency
    ):
        summary += msg.choices[0].delta.content

    return summary.strip()

def respond(message, history, system_message, max_tokens, temperature, top_p):
    if history is None:
        history = []

    # Summarize conversation if history is long
    if len(history) >= SUMMARIZE_AFTER:
        summary = summarize_conversation(history[:SUMMARIZE_AFTER])  # Summarize only the first N messages
        history = [(f"Summary: {summary}", "")] + history[-MAX_HISTORY:]  # Keep summary + latest history

    # Format conversation for Zephyr-7B
    formatted_history = [{"role": "system", "content": system_message}]

    for user_msg, bot_reply in history:
        formatted_history.append({"role": "user", "content": user_msg})
        formatted_history.append({"role": "assistant", "content": bot_reply})

    formatted_history.append({"role": "user", "content": message})

    # Get response from Zephyr-7B
    response = ""
    for msg in client.chat_completion(
        formatted_history,  
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = msg.choices[0].delta.content
        response += token
        yield response

    history.append((message, response))  # Maintain history

    return history  # Return updated history

# Create Gradio ChatInterface
interface = gr.ChatInterface(
    fn=respond,
    title="Mistral-7B Chatbot 🤖",
    description="💬 Chat with a fine-tuned Mistral-7B model for interactive conversations.",
    theme="soft",
    additional_inputs=[
        gr.Textbox(value="You are a friendly chatbot.", label="🛠 System Message", interactive=True),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="📏 Max Tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="🔥 Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="🎯 Top-p Sampling"),
    ],
)

interface.launch(share=True)
