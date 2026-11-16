# GraphTopic-Topic-Modeling

A simple Python project for learning about topic modeling.

## What This Does
This project helps you understand how computers can find topics in groups of text.

## How It Works
1. You give it some text
2. It looks for patterns
3. It groups similar words together

## Getting Started
```python
# Simple example
from topic_model import TopicFinder
finder = TopicFinder()
finder.analyze(["hello world", "test data"])
```

## Learn More
- Search online for "introduction to topic modeling"
- Try different text inputs
- See what patterns emerge

## Built With
- Python
- Basic text analysis

## Author
Made as a learning project

## Faster Web Demo
The web app uses a deterministic, label-balanced 2,000-document subset of the 12,000-document
corpus for its quick presentation demo. It caches the computed sample analysis for repeat runs in
the same process and creates charts only when the technical-results section is opened. Uploaded
and pasted corpora continue to use the full provided input.
