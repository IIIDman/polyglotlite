"""
Language Detection and Utilities

Provides language detection and multilingual support utilities.
"""

from typing import Dict, List, Optional, Tuple
import re
from collections import Counter


# Language character ranges and patterns
LANGUAGE_PATTERNS = {
    # Latin-based languages (need additional heuristics)
    'en': {
        'common_words': ['the', 'is', 'are', 'was', 'been', 'have', 'has', 'will', 'would', 'could'],
        'stopwords': ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for'],
    },
    'es': {
        'common_words': ['el', 'la', 'los', 'las', 'es', 'son', 'está', 'están', 'que', 'de'],
        'stopwords': ['el', 'la', 'de', 'que', 'y', 'en'],
        'chars': 'áéíóúüñ',
    },
    'fr': {
        'common_words': ['le', 'la', 'les', 'est', 'sont', 'était', 'que', 'qui', 'dans', 'pour'],
        'stopwords': ['le', 'la', 'de', 'et', 'en', 'un'],
        'chars': 'àâçéèêëïîôùûü',
    },
    'de': {
        'common_words': ['der', 'die', 'das', 'ist', 'sind', 'war', 'haben', 'werden', 'nicht'],
        'stopwords': ['der', 'die', 'und', 'in', 'zu'],
        'chars': 'äöüß',
    },
    'pt': {
        'common_words': ['o', 'a', 'os', 'as', 'é', 'são', 'está', 'estão', 'que', 'de'],
        'stopwords': ['o', 'a', 'de', 'que', 'e'],
        'chars': 'àáâãçéêíóôõú',
    },
    'it': {
        'common_words': ['il', 'lo', 'la', 'è', 'sono', 'che', 'per', 'non', 'con', 'una'],
        'stopwords': ['il', 'la', 'di', 'che', 'e'],
        'chars': 'àèéìòù',
    },
    'nl': {
        'common_words': ['de', 'het', 'een', 'is', 'zijn', 'van', 'dat', 'die', 'niet', 'voor'],
        'stopwords': ['de', 'het', 'van', 'en', 'een'],
    },
    'pl': {
        'common_words': ['jest', 'są', 'to', 'nie', 'się', 'na', 'do', 'że', 'jak', 'ale'],
        'stopwords': ['i', 'w', 'nie', 'na', 'do'],
        'chars': 'ąćęłńóśźż',
    },
    'ru': {
        'range': (0x0400, 0x04FF),  # Cyrillic
        'common_words': ['и', 'в', 'не', 'на', 'что', 'он', 'как', 'это', 'она', 'они'],
    },
    'uk': {
        'range': (0x0400, 0x04FF),  # Cyrillic
        'chars': 'іїєґ',
    },
    'ar': {
        'range': (0x0600, 0x06FF),  # Arabic
    },
    'fa': {
        'range': (0x0600, 0x06FF),  # Arabic (Persian uses Arabic script)
        'chars': 'پچژگ',
    },
    'hi': {
        'range': (0x0900, 0x097F),  # Devanagari
    },
    'bn': {
        'range': (0x0980, 0x09FF),  # Bengali
    },
    'ta': {
        'range': (0x0B80, 0x0BFF),  # Tamil
    },
    'te': {
        'range': (0x0C00, 0x0C7F),  # Telugu
    },
    'th': {
        'range': (0x0E00, 0x0E7F),  # Thai
    },
    'vi': {
        'chars': 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ',
    },
    'zh': {
        'range': (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    },
    'ja': {
        'ranges': [(0x3040, 0x309F), (0x30A0, 0x30FF)],  # Hiragana + Katakana
    },
    'ko': {
        'range': (0xAC00, 0xD7AF),  # Korean Hangul
    },
    'he': {
        'range': (0x0590, 0x05FF),  # Hebrew
    },
    'el': {
        'range': (0x0370, 0x03FF),  # Greek
    },
    'tr': {
        'common_words': ['bir', 've', 'bu', 'için', 'ile', 'de', 'da', 'ben', 'sen', 'o'],
        'chars': 'çğıöşü',
    },
    'id': {
        'common_words': ['yang', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk', 'adalah', 'pada'],
        'stopwords': ['yang', 'dan', 'di', 'ke', 'dari'],
    },
    'ms': {
        'common_words': ['yang', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk', 'adalah', 'pada'],
    },
    'sw': {
        'common_words': ['na', 'wa', 'ya', 'ni', 'kwa', 'katika', 'hii', 'hilo', 'yake'],
    },
}

# Supported languages with their full names
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'pt': 'Portuguese',
    'it': 'Italian',
    'nl': 'Dutch',
    'pl': 'Polish',
    'ru': 'Russian',
    'uk': 'Ukrainian',
    'ar': 'Arabic',
    'fa': 'Persian',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'he': 'Hebrew',
    'el': 'Greek',
    'tr': 'Turkish',
    'id': 'Indonesian',
    'ms': 'Malay',
    'sw': 'Swahili',
    # More languages...
    'cs': 'Czech',
    'sk': 'Slovak',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
    'hr': 'Croatian',
    'sr': 'Serbian',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'fi': 'Finnish',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'da': 'Danish',
    'is': 'Icelandic',
    'ga': 'Irish',
    'cy': 'Welsh',
    'eu': 'Basque',
    'ca': 'Catalan',
    'gl': 'Galician',
    'af': 'Afrikaans',
    'zu': 'Zulu',
    'xh': 'Xhosa',
    'yo': 'Yoruba',
    'ig': 'Igbo',
    'ha': 'Hausa',
    'am': 'Amharic',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'my': 'Burmese',
    'km': 'Khmer',
    'lo': 'Lao',
    'mn': 'Mongolian',
    'ka': 'Georgian',
    'hy': 'Armenian',
    'az': 'Azerbaijani',
    'kk': 'Kazakh',
    'uz': 'Uzbek',
    'tg': 'Tajik',
    'ur': 'Urdu',
    'pa': 'Punjabi',
    'gu': 'Gujarati',
    'mr': 'Marathi',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'or': 'Odia',
    'as': 'Assamese',
}


def detect_language(text: str, return_scores: bool = False) -> str:
    """
    Detect the language of input text.
    Returns ISO 639-1 language code (e.g., 'en', 'es', 'zh')
    """
    if not text or not text.strip():
        return 'en'  # Default to English
    
    text = text.strip()
    scores = Counter()
    
    # Check character-based languages first - these are most reliable
    # FIXME: should probably weight CJK characters higher
    char_counts = Counter()
    for char in text:
        code = ord(char)
        for lang, pattern in LANGUAGE_PATTERNS.items():
            if 'range' in pattern:
                start, end = pattern['range']
                if start <= code <= end:
                    char_counts[lang] += 1
            if 'ranges' in pattern:
                for start, end in pattern['ranges']:
                    if start <= code <= end:
                        char_counts[lang] += 1
    
    # Strong signal from character ranges
    if char_counts:
        dominant_lang = char_counts.most_common(1)[0]
        if dominant_lang[1] > len(text) * 0.3:  # >30% of chars
            if return_scores:
                return {dominant_lang[0]: 1.0}
            return dominant_lang[0]
    
    # Check for specific characters (Vietnamese, Turkish, etc.)
    for lang, pattern in LANGUAGE_PATTERNS.items():
        if 'chars' in pattern:
            for char in pattern['chars']:
                if char.lower() in text.lower():
                    scores[lang] += 5
    
    # Word-based detection for Latin scripts
    words = re.findall(r'\b\w+\b', text.lower())
    if words:
        for lang, pattern in LANGUAGE_PATTERNS.items():
            if 'common_words' in pattern:
                matches = sum(1 for w in words if w in pattern['common_words'])
                scores[lang] += matches * 10
            if 'stopwords' in pattern:
                matches = sum(1 for w in words if w in pattern['stopwords'])
                scores[lang] += matches * 5
    
    # Default to English if no strong signal
    if not scores:
        scores['en'] = 1
    
    if return_scores:
        total = sum(scores.values())
        return {lang: score/total for lang, score in scores.items()}
    
    return scores.most_common(1)[0][0]


def get_supported_languages() -> Dict[str, str]:
    """
    Get dictionary of supported languages.
    
    Returns:
        Dict mapping language codes to language names
    """
    return SUPPORTED_LANGUAGES.copy()


def is_language_supported(lang_code: str) -> bool:
    """Check if a language code is supported."""
    return lang_code.lower() in SUPPORTED_LANGUAGES


def get_language_name(lang_code: str) -> str:
    return SUPPORTED_LANGUAGES.get(lang_code.lower(), "Unknown")


def detect_script(text: str) -> str:
    """
    Detect the writing script of text.
    
    Returns one of: 'latin', 'cyrillic', 'arabic', 'devanagari', 
    'cjk', 'hangul', 'hebrew', 'greek', 'thai', 'other'
    """
    if not text:
        return 'other'
    
    scripts = Counter()
    
    for char in text:
        code = ord(char)
        if 0x0041 <= code <= 0x024F:
            scripts['latin'] += 1
        elif 0x0400 <= code <= 0x04FF:
            scripts['cyrillic'] += 1
        elif 0x0600 <= code <= 0x06FF:
            scripts['arabic'] += 1
        elif 0x0900 <= code <= 0x097F:
            scripts['devanagari'] += 1
        elif 0x4E00 <= code <= 0x9FFF:
            scripts['cjk'] += 1
        elif 0x3040 <= code <= 0x30FF:
            scripts['japanese'] += 1
        elif 0xAC00 <= code <= 0xD7AF:
            scripts['hangul'] += 1
        elif 0x0590 <= code <= 0x05FF:
            scripts['hebrew'] += 1
        elif 0x0370 <= code <= 0x03FF:
            scripts['greek'] += 1
        elif 0x0E00 <= code <= 0x0E7F:
            scripts['thai'] += 1
    
    if scripts:
        return scripts.most_common(1)[0][0]
    return 'other'


def normalize_text(text: str, lang: Optional[str] = None) -> str:
    """
    Normalize text for processing.
    
    Args:
        text: Input text
        lang: Optional language code for language-specific normalization
        
    Returns:
        Normalized text
    """
    import unicodedata
    
    # Normalize unicode
    text = unicodedata.normalize('NFC', text)
    
    # Remove control characters
    text = ''.join(char for char in text if unicodedata.category(char) != 'Cc')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text
