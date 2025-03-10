A lightweight, optimized **Mistral-7B** chatbot with **8-bit quantization (BitsAndBytes)**, **PyTorch compilation**, and a **Gradio UI**. This chatbot is designed for **efficient inference on limited hardware** while maintaining conversational context using **summarization techniques**.

## 🔥 Features
✅ **8-bit quantization** (with BitsAndBytes) for **low-VRAM** usage.  
✅ **PyTorch compilation (`torch.compile()`)** for **faster inference**.  
✅ **Context-aware chat history** with conversation **summarization**.   
✅ **Gradio UI** for a seamless chatbot experience.  
✅ **Optimized GPU performance with `torch.backends.cudnn.benchmark = True`**.  

---

## 📌 Installation & Setup

### **1️⃣ Clone the Repository**

- git clone https://github.com/manojrathor08/Mistral-7B-Chatbot.git
- cd Mistral-7B-Chatbot

### **2️⃣ Install Dependencies**
Ensure you have **Python 3.9+** and install the required libraries:


- pip install torch transformers gradio huggingface_hub bitsandbytes

### **3️⃣ Run the Chatbot**

- python3 app.py

The chatbot will launch locally. You’ll see a link like:

Running on local URL:  http://127.0.0.1:7860

Running on public URL:  https://your-gradio-link.gradio.live

Click on the link to interact with the chatbot! 🚀

## ⚙️ Model & Optimization Details

### **Mistral-7B Model**
- The chatbot runs on **Mistral-7B-Instruct**, a **powerful open-weight LLM**.
- Uses **8-bit quantization (`BitsAndBytesConfig`)** for **lower memory usage**.
- Keeps **final layers (`lm_head`) in float16** for **better response quality**.

### **Speed Optimizations**
- **Quantization** (`load_in_8bit=True`) reduces memory usage.
- **PyTorch compilation** (`torch.compile()`) accelerates inference.
- **Uses `torch.no_grad()`** to disable gradient calculations for inference.
- **`torch.backends.cudnn.benchmark = True`** improves CUDA performance.
# 🎨 UI & Chat Memory Handling

### 🗣️ Multi-turn Conversation Support
- Stores **last 2 interactions** (`MAX_HISTORY = 2`) for maintaining context.
- Uses a **summarization mechanism** (`SUMMARIZE_AFTER = 3`) to condense long conversations.

## 💬 Gradio UI

### **Title**  
📝 **"Mistral-7B Chatbot 🤖"**

### **Customizable Settings**
- **System message:** `"You are a friendly chatbot."`
- **Max tokens:** `1-2048`
- **Temperature:** `0.1-4.0`
- **Top-p Sampling:** `0.1-1.0`



## 🛠️ How It Works
- **Processes recent chat history** and formats it properly.
- **If history gets too long**, it **summarizes past messages** to maintain context.
- **Streams real-time responses** to improve user experience.

## 📜 Example Conversation

![Chatbot Conversation](images/chatbot_image.png)

## 📌 Contributing
Want to improve the chatbot? **Contributions are welcome!** 🚀

1. **Fork the repo**  
2. **Create a new branch** (`feature-improvement`)  
3. **Commit and push changes**  
4. **Open a pull request**  


## 📜 License
This project is **MIT Licensed** – feel free to **modify and use it!**  

⭐ **Star this repo** if you found it useful! 😊
