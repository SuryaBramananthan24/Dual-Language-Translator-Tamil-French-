# 🌍 Dual-Language Translator (Tamil ↔ French)

A simple Python-based application that translates **any input text** into **Tamil** and **French** simultaneously using **LangChain** and **OpenAI GPT models**.

---

## ✨ Features

- 🌐 Translates text into **Tamil** and **French** at the same time  
- ⚡ Uses **LangChain RunnableParallel** for parallel translation  
- 🖥️ Simple **console-based** interface  
- 🧠 Easy to understand and extend  

---

## 🛠️ Technologies Used

- **Python**
- **LangChain**
- **OpenAI (GPT-4o-mini)**
- **RunnableParallel**

---

## 📂 Project Workflow

1. The user enters a sentence (any language)
2. The text is sent to the AI model
3. Translations run in parallel
4. Tamil and French translations are displayed

---

## 📌 Requirements

- Python **3.8+**
- OpenAI API Key
- Active internet connection

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SuryaBramananthan24/Dual-Language-Translator-Tamil-French-
cd Dual-Language-Translator-Tamil-French-
```
### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
```
 Activate the Virtual Environment

Linux / macOS
```bash
source venv/bin/activate
```
Windows
```bash
venv\Scripts\activate
```
### 3️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Set the OpenAI API Key

Linux / macOS
```bash
export OPENAI_API_KEY="your_api_key_here"
```
Windows (PowerShell)
```bash
$env:OPENAI_API_KEY="your_api_key_here"
```

---

## ▶️ How to Run the Project
```bash
python translate.py
```
📥 Input
Enter any sentence when prompted.

📤 Output
The program prints:
Tamil translation
French translation

---

## 📈 Future Enhancements

- Add support for more languages
- Build a web or GUI interface
- Add speech-to-text and text-to-speech
- Improve error handling and validation
