# AI Agents for Economic Market Analysis
This project is to create agents to do market research on the US Economy.  It will review Consumers, Government, Investments, and Net Exports

# Hompage View
![Model](https://github.com/kvongrassamy/EconMarketAnalysisAI/blob/main/image/EconHomePage.PNG)

# AI Agent Team!
- Market Researcher:  It will review news articles related on econmics industries listed below and collect information on each industry:
    -Healthcare
    -Investments
    -Technology
    -Finance
    -Construction
    -Real Estate
- Economist:  It will take the results from these news articles and provide economic context what might cause these issues or why its an issue in the economy
- Evaluator:  It will provide information on the industry and review economic textbooks to resolve these issues at a high level


SETUP
- Create vnenv (Python Version: 3.13):
```bash 
py -3.13 -m venv econanalysis
```

- activate the env with 
```bash 
source ./econanalysis/Scripts/activate
```

- install requirements: 
```bash
pip install -r requirements.txt
```

- API Keys Are required: 
    - See sample .env file in .env_sample


- Run the Streamlit App: 
```bash
streamlit run home.py
```
