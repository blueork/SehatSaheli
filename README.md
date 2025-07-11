# SehatSaheli
Demo

### How to run it

1. Create a virtual environment. I have used conda.
```conda create --name <env-name> python=3.10```

2. Activate the conda virtual environment.
```conda activate <env-name>```

3. Install packages from requirements.txt
```pip install -r requirements.txt```

4. Add your HUGGINGFACEHUB_API_TOKEN in the demo.py file.
You can generate a free token from Hugging Face after signing up
https://huggingface.co/docs/hub/en/security-tokens

5. Run the application
```streamlit run demo.py```
