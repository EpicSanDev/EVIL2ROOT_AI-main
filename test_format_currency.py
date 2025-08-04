from flask import Flask
from src.utils.filters import register_filters

def test_format_currency_preserves_negative_sign():
    app = Flask(__name__)
    register_filters(app)
    fmt = app.jinja_env.filters['format_currency']
    result = fmt('-1234.56')
    assert result.startswith('-') or result.startswith('(')
