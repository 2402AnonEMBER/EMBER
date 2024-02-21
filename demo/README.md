## Running the demo
Installation:
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```
Running the app:
```
streamlit run app.py
```
By default, this will run the demo with classifiers trained on generated text, annotated according to CoNLL2003. To run with Ontonotes classifier:
```
streamlit run app.py -- --dataset tner/ontonotes5
```
To run with CoNLL2003 classifier (not trained on generated data):
```
streamlit run app.py -- --dataset conll2003
```

## Running with Docker (slow, not recommended)

```
docker build -t ember_demo .
docker run -p 8501:8501 -v /path/to/host/cache:/app/cache ember_demo
```

By default, this will run the demo with classifiers trained on generated text, annotated according to CoNLL2003.

To run with Ontonotes classifier
```
docker run -p 8501:8501 -v /path/to/host/cache:/app/cache ember_demo streamlit run app.py -- --dataset tner/ontonotes5
```

To run with CoNLL2003 classifier (not trained on generated data)
```
docker run -p 8501:8501 -v /path/to/host/cache:/app/cache ember_demo streamlit run app.py -- --dataset conll2003
```