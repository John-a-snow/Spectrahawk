install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

app:
	streamlit run app.py

demo: install test app
