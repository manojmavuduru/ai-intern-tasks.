# AI/ML Internship Project Portfolio

Five self-contained AI/ML projects, each demonstrating a different
core technique — from rule-based logic and classic search algorithms
through to deep learning with CNNs, RNNs, and metric learning. Every
task includes working code, a Streamlit demo UI, unit tests, and its
own README with setup instructions and extension ideas.

| # | Task | Core Techniques | Demo |
|---|------|------------------|------|
| 1 | [Rule-Based Chatbot](task1-rule-based-chatbot/) | Regex pattern matching, intent rules | `streamlit run app.py` |
| 2 | [Tic-Tac-Toe AI](task2-tictactoe-ai/) | Minimax, Alpha-Beta Pruning, game theory | `streamlit run app.py` |
| 3 | [Image Captioning](task3-image-captioning/) | CNN (ResNet/VGG) + LSTM, transfer learning | `streamlit run app.py` |
| 4 | [Recommendation System](task4-recommendation-system/) | TF-IDF, collaborative filtering, SVD | `streamlit run app.py` |
| 5 | [Face Detection & Recognition](task5-face-detection-recognition/) | Haar cascades, Siamese networks, triplet loss | `streamlit run app.py` |

## Repository structure

```
.
├── task1-rule-based-chatbot/
├── task2-tictactoe-ai/
├── task3-image-captioning/
├── task4-recommendation-system/
├── task5-face-detection-recognition/
├── .gitignore
├── LICENSE
└── README.md   <- you are here
```

Each task folder is **fully independent** — its own `requirements.txt`,
its own `README.md`, its own tests. You can clone the whole repo and
run any single task without needing the others installed.

## Quick start (any task)

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>/task1-rule-based-chatbot   # or any task folder
pip install -r requirements.txt
streamlit run app.py
```

## Why these five, together

This set intentionally spans a gradient of AI/ML complexity:

1. **Rule-based logic** (no ML at all) — the foundation of conversational systems.
2. **Classical search/game theory** — Minimax shows how "intelligence" can emerge from exhaustive search, no training data required.
3. **Deep learning, multi-modal** — combining computer vision (CNN) and NLP (RNN) in an encoder-decoder pipeline.
4. **Classical ML for personalization** — TF-IDF similarity and matrix factorization, the backbone of real-world recommender systems.
5. **Computer vision + metric learning** — object detection paired with a from-scratch Siamese network, the same family of technique behind production face-recognition systems.

Each README explains *why* specific design choices were made (not
just *how* to run the code), what each project's limitations are,
and concrete ways to extend it further — useful both for
demonstrating understanding and as a roadmap for follow-up work.

## Testing

Every task includes a unit test suite. From inside any task folder:
```bash
python -m unittest discover -p "test_*.py" -v
```
All test suites pass as of the latest commit.

## Deploying the demos

Each Streamlit app can be deployed for free on
[Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repo to GitHub (see below).
2. On Streamlit Cloud, click "New app", point it at your repo, and set the **main file path** to e.g. `task1-rule-based-chatbot/app.py`.
3. Repeat per task (each needs its own Streamlit app deployment, since they're independent projects with separate `requirements.txt` files).

## License

MIT — see [LICENSE](LICENSE).
