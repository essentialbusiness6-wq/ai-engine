from app.services.text_normalizer import RuleBasedTextNormalizer


def test_expands_abbreviations():
    n = RuleBasedTextNormalizer()
    assert n.normalize("send inv pls") == "send invoice please"


def test_corrects_common_misspelling():
    n = RuleBasedTextNormalizer()
    # "invoic" is 1 edit away from "invoice"
    result = n.normalize("show my invoic status")
    assert "invoice" in result


def test_preserves_invoice_codes_untouched():
    n = RuleBasedTextNormalizer()
    result = n.normalize("pay INV-2024-001 now")
    assert "INV-2024-001" in result


def test_collapses_whitespace():
    n = RuleBasedTextNormalizer()
    result = n.normalize("  hello    world  ")
    assert result == "hello world"


def test_preserves_trailing_punctuation_on_expansion():
    n = RuleBasedTextNormalizer()
    result = n.normalize("do it asap!")
    assert result == "do it as soon as possible!"


def test_does_not_mangle_numbers():
    n = RuleBasedTextNormalizer()
    result = n.normalize("pay 5000 now")
    assert "5000" in result


def test_custom_abbreviations_merge_with_defaults():
    n = RuleBasedTextNormalizer(abbreviations={"po": "purchase order"})
    result = n.normalize("check po and inv")
    assert "purchase order" in result and "invoice" in result
