# Ollamalit

This is a Ollamalit.

## Cloning

1. Clone the repository:

   ```bash
   git clone https://github.com/chungchihhan/ollamalit.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ollamalit
   ```

## Virtual Environment

3. Create a virtual environment:

   ```bash
   python3 -m venv venv
   ```

4. Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

## Installation

5. Install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

## Start the application

6. (Optional)Prepare your data:

   - put you some pdf files you want to read in the folder named `rag-files`
   - put gguf files that you have downloaded in the folder named `gguf-files`, you can dowmload gguf files from huggingface.

7. Run the command in the terminal:

   ```bash
   streamlit run hello.py
   ```
