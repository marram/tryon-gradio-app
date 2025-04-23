# Simple Virtual Try-On App using FASHN AI & Gradio

This repository implements a simple virtual try-on app that uses the Gradio framework and the FASHN API as its backend.

<p align="center">
    <img src="./assets/screenshot.png" alt="app screenshot">
</p>

### Sign Up to FASHN
This repository requires an API key from a FASHN account.
Don't have an account yet? [Create an account](https://app.fashn.ai/).

If you already have an account, go to Settings → API → `+ Create new API key`

### Setup

1. Clone this repository:
```bash
git clone https://github.com/fashn-AI/tryon-gradio-app
cd tryon-gradio-app
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. Set the environment variable `FASHN_API_KEY` to the API key you created on the FASHN platform.
```bash
export FASHN_API_KEY="your-api-key"
```
2. Run the app:
```bash
python app.py
```

3. Open your browser and go to `http://localhost:7860/` to see the app in action.

If you wish to deploy the app or share it with others, you can use the `share` function in Gradio to generate a public link.
More on this can be found in the [Gradio documentation](https://gradio.app/docs).

### Helpful Guides and Documentation

To get the most out of the FASHN API, we recommend to read the following guides to better understand all node features and parameters:
1. [API Parameters Guide](https://docs.fashn.ai/guides/api-parameters-guide)
2. [Official API Docs](https://docs.fashn.ai/fashn-api/endpoints#request)

