from datetime import datetime, timezone
from decimal import Decimal

from app.domain.entities import EntityType
from app.services.entity_extractor import RegexEntityExtractor


def test_extracts_invoice_number():
    e = RegexEntityExtractor()
    result = e.extract("please check invoice #12345")
    invoice_entities = [x for x in result if x.type == EntityType.INVOICE_NUMBER]
    assert len(invoice_entities) == 1
    assert invoice_entities[0].value == "12345"


def test_extracts_payment_reference():
    e = RegexEntityExtractor()
    result = e.extract("my transaction ref TXN9988ABC")
    refs = [x for x in result if x.type == EntityType.PAYMENT_REFERENCE]
    assert len(refs) >= 1


def test_extracts_amount_with_dollar_symbol():
    e = RegexEntityExtractor()
    result = e.extract("pay $500.50 now")
    amounts = [x for x in result if x.type == EntityType.AMOUNT]
    currencies = [x for x in result if x.type == EntityType.CURRENCY]
    assert amounts[0].value == Decimal("500.50")
    assert currencies[0].value == "USD"


def test_extracts_amount_with_currency_code():
    e = RegexEntityExtractor()
    result = e.extract("send 2000 NGN to the vendor")
    amounts = [x for x in result if x.type == EntityType.AMOUNT]
    currencies = [x for x in result if x.type == EntityType.CURRENCY]
    assert amounts[0].value == Decimal("2000")
    assert currencies[0].value == "NGN"


def test_bare_number_without_currency_is_not_amount():
    e = RegexEntityExtractor()
    result = e.extract("call me at 5")
    amounts = [x for x in result if x.type == EntityType.AMOUNT]
    assert amounts == []


def test_extracts_relative_date_tomorrow():
    ref = datetime(2026, 7, 27, tzinfo=timezone.utc)
    e = RegexEntityExtractor(reference_time=ref)
    result = e.extract("remind me tomorrow")
    dates = [x for x in result if x.type == EntityType.RELATIVE_DATE]
    assert len(dates) == 1
    assert dates[0].value.date() == datetime(2026, 7, 28).date()


def test_extracts_relative_date_in_n_days():
    ref = datetime(2026, 7, 27, tzinfo=timezone.utc)
    e = RegexEntityExtractor(reference_time=ref)
    result = e.extract("schedule it in 5 days")
    dates = [x for x in result if x.type == EntityType.RELATIVE_DATE]
    assert dates[0].value.date() == datetime(2026, 8, 1).date()


def test_extracts_absolute_date():
    e = RegexEntityExtractor()
    result = e.extract("the invoice was issued on 2024-03-12")
    dates = [x for x in result if x.type == EntityType.DATE]
    assert len(dates) == 1
    assert dates[0].value.year == 2024
    assert dates[0].value.month == 3


def test_extracts_email():
    e = RegexEntityExtractor()
    result = e.extract("send the receipt to jane.doe@example.com")
    emails = [x for x in result if x.type == EntityType.EMAIL]
    assert emails[0].value == "jane.doe@example.com"


def test_extracts_person_name_heuristically():
    e = RegexEntityExtractor()
    result = e.extract("Please pay John Smith for the consulting work")
    names = [x for x in result if x.type == EntityType.PERSON_NAME]
    assert any(n.value == "John Smith" for n in names)


def test_does_not_treat_sentence_start_stopword_as_name():
    e = RegexEntityExtractor()
    result = e.extract("Show me the invoice")
    names = [x for x in result if x.type == EntityType.PERSON_NAME]
    assert names == []
