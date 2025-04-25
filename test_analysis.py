from main import preprocess, top20

def test_preprocess():
    assert preprocess("Hello, world!") == ["hello", "world"]

def test_top20():
    data = "a b a".split()
    assert top20(data)[0] == ("a", 2)