## Conda Related

#### List all the environments created with conda
1. Open the Command Prompt
2. Run the following command
```
conda env list
```

#### Create a conda environment
1. Open the Command Prompt
2. Run the following command
```
conda create --name myenv python=3.10
```
Replace `myenv` with your desired environment name, and `3.10` with the desired Python version.

#### Activate a conda environment
1. Open the Command Prompt
2. Run the following command
```
conda activate <env-name>
```

#### Deactivate a conda environment
1. Open the Command Prompt
2. Run the following command
```
conda deactivate
```

#### Remove a conda environment
1. Open the Command Prompt
2. Run the following command
```
conda remove --name myenv --all
```
Replace `myenv` with your desired environment name.

#### To see the list of installed packages in your active Conda environment
1. Activate your conda environment
2. Run the following command
```
conda list
```

#### Install packages from requirements.txt
1. If you want to install the packges in your conda environment, then first activate the environment and the follow. Otherwise skip to the second step.
2. Open Command Prompt
3. Run the following command
```
pip install -r requirements.txt
```

## Streamlit Related

#### Run a streamlit application
```
streamlit run app.py
```
Replace `app` with the relevant file name

## Ollama Related

#### To locally host a particular LLM
1. Ensure you have ollama installed
2. Open the Command Prompt
3. Run the following command
```
ollama run <model-name>
```

#### To see the models already downloaded on your device
1. Ensure you have ollama installed
2. Open the Command Prompt
3. Run the following command
```
ollama list
```


