from unchain.cli.repl import route_input


def test_route_input_prefixes():
    assert route_input("/btw how long?") == ("btw", "how long?")
    assert route_input("/fyi also support Chinese") == ("fyi", "also support Chinese")
    assert route_input("/steer next refactor the tests") == ("steer", "next refactor the tests")


def test_route_input_bare_text_is_unknown():
    assert route_input("hello there") == ("unknown", "hello there")


def test_route_input_empty_and_whitespace():
    assert route_input("") == ("empty", "")
    assert route_input("   ") == ("empty", "")
    assert route_input("/fyi   ") == ("empty", "")
