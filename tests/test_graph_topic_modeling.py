import pytest
from graph_topic_modeling import TopicModeler

def test_basic_functionality():
    """Test basic topic modeling functionality."""
    model = TopicModeler()
    texts = ["This is a test document about machine learning", "Another document about artificial intelligence"]
    model.fit(texts)
    results = model.transform(texts)
    assert len(results) == 2

def test_empty_texts():
    """Test handling of empty texts."""
    model = TopicModeler()
    with pytest.raises(ValueError):
        model.fit([])