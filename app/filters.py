from flask import Blueprint
from datetime import datetime
import locale
import re

# Set locale for currency formatting
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    # Fallback if the locale is not available
    locale.setlocale(locale.LC_ALL, '')

def register_filters(app):
    """Register custom Jinja2 filters for the app."""
    
    @app.template_filter('format_currency')
    def format_currency(value):
        """Format a value as currency, handling strings or numbers."""
        if isinstance(value, str):
            # Remove any existing currency symbols and commas
            value = re.sub(r'[^\d.]', '', value)
            try:
                value = float(value)
            except ValueError:
                return value  # Return original if conversion fails
        
        if isinstance(value, (int, float)):
            try:
                return locale.currency(value, grouping=True)
            except:
                # Fallback if locale formatting fails
                return f"{value:,.2f}"
        
        return value  # Return original if not a number
    
    @app.template_filter('format_timestamp')
    def format_timestamp(timestamp):
        """Format an ISO timestamp to a human-readable date and time."""
        if not timestamp:
            return "-"
        
        try:
            if isinstance(timestamp, str):
                # Parse ISO format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
                
            # Format as a readable string
            return dt.strftime("%b %d, %Y %H:%M")
        except Exception:
            return timestamp  # Return original if parsing fails
    
    @app.template_filter('format_percent')
    def format_percent(value, decimals=2):
        """Format a value as a percentage with the specified number of decimal places."""
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return value
        
        if isinstance(value, (int, float)):
            return f"{value:.{decimals}f}%"
        
        return value
    
    @app.template_filter('truncate_address')
    def truncate_address(address, length=10):
        """Truncate a crypto address to a shorter form (start...end)."""
        if not address or not isinstance(address, str):
            return address
        
        if len(address) <= length * 2:
            return address
        
        return f"{address[:length]}...{address[-length:]}"
    
    @app.template_filter('time_ago')
    def time_ago(timestamp):
        """Convert a timestamp to a 'time ago' format."""
        if not timestamp:
            return "-"
        
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            
            now = datetime.now()
            diff = now - dt
            
            seconds = diff.total_seconds()
            
            if seconds < 60:
                return f"{int(seconds)} seconds ago"
            elif seconds < 3600:
                return f"{int(seconds // 60)} minutes ago"
            elif seconds < 86400:
                return f"{int(seconds // 3600)} hours ago"
            elif seconds < 604800:
                return f"{int(seconds // 86400)} days ago"
            elif seconds < 2592000:
                return f"{int(seconds // 604800)} weeks ago"
            elif seconds < 31536000:
                return f"{int(seconds // 2592000)} months ago"
            else:
                return f"{int(seconds // 31536000)} years ago"
                
        except Exception:
            return timestamp  # Return original if parsing fails
    
    @app.template_filter('format_number')
    def format_number(value, decimals=0):
        """Format a number with thousands separators and specified decimal places."""
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return value
        
        if isinstance(value, (int, float)):
            return f"{value:,.{decimals}f}"
        
        return value
    
    @app.template_filter('color_class')
    def color_class(value, neutral_threshold=0.05):
        """Return a CSS class based on the value being positive, negative, or neutral."""
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return ""
        
        if isinstance(value, (int, float)):
            if value > neutral_threshold:
                return "positive"
            elif value < -neutral_threshold:
                return "negative"
            else:
                return "neutral"
        
        return "" 