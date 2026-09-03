from app.services.intent_recognizer import TfidfIntentRecognizer


def test_recognizes_invoice_status_intent():
    r = TfidfIntentRecognizer()
    result = r.recognize("what is the status of invoice 456")
    assert result.intent == "get_invoice_status"
    assert result.confidence > 0.25


def test_recognizes_payment_intent():
    r = TfidfIntentRecognizer()
    result = r.recognize("i want to pay this invoice now")
    assert result.intent == "create_payment"


def test_recognizes_greeting():
    r = TfidfIntentRecognizer()
    result = r.recognize("hello")
    assert result.intent == "greeting"


def test_empty_text_is_unknown():
    r = TfidfIntentRecognizer()
    result = r.recognize("")
    assert result.intent == "unknown"
    assert result.confidence == 0.0


def test_gibberish_falls_back_to_unknown_with_low_confidence():
    r = TfidfIntentRecognizer()
    result = r.recognize("purple elephant quantum banana")
    assert result.intent == "unknown"


def test_alternatives_are_populated():
    r = TfidfIntentRecognizer()
    result = r.recognize("show me unpaid invoices")
    assert isinstance(result.alternatives, list)
