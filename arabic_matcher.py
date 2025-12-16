"""
Arabic text normalization and matching utilities
Handles variations in Arabic characters (ة/ه, ا/أ, etc.) and diacritics
"""
import re
import unicodedata
from typing import Tuple

def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text by:
    1. Removing diacritics (harakat)
    2. Normalizing alef variations (ا, أ, إ, آ)
    3. Normalizing ta marbuta (ة) to ha (ه)
    4. Removing extra whitespace
    5. Normalizing to NFD and removing combining marks
    """
    if not text:
        return ""
    
    # Remove diacritics (harakat) - combining marks
    text = ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )
    
    # Normalize alef variations (ا, أ, إ, آ) to ا
    alef_variations = ['أ', 'إ', 'آ', 'ا']
    for alef in alef_variations:
        text = text.replace(alef, 'ا')
    
    # Normalize ta marbuta (ة) to ha (ه)
    text = text.replace('ة', 'ه')
    
    # Normalize yeh variations (ي, ى)
    text = text.replace('ى', 'ي')
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', '', text.strip())
    
    return text

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two Arabic texts (0.0 to 1.0)
    Uses normalized text and character-based similarity
    """
    norm1 = normalize_arabic_text(text1)
    norm2 = normalize_arabic_text(text2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Exact match after normalization
    if norm1 == norm2:
        return 1.0
    
    # Calculate character-based similarity (Jaccard-like)
    set1 = set(norm1)
    set2 = set(norm2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    jaccard_similarity = intersection / union
    
    # Also check substring matching
    if norm1 in norm2 or norm2 in norm1:
        # If one is substring of another, boost similarity
        min_len = min(len(norm1), len(norm2))
        max_len = max(len(norm1), len(norm2))
        substring_boost = min_len / max_len
        return max(jaccard_similarity, substring_boost * 0.9)
    
    # Check for common prefix/suffix
    common_prefix = 0
    common_suffix = 0
    
    min_len = min(len(norm1), len(norm2))
    for i in range(min_len):
        if norm1[i] == norm2[i]:
            common_prefix += 1
        else:
            break
    
    for i in range(1, min_len + 1):
        if norm1[-i] == norm2[-i]:
            common_suffix += 1
        else:
            break
    
    # Weighted similarity
    prefix_weight = common_prefix / max(len(norm1), len(norm2))
    suffix_weight = common_suffix / max(len(norm1), len(norm2))
    
    return max(jaccard_similarity, (prefix_weight + suffix_weight) / 2)

def is_match(target: str, transcription: str, threshold: float = 0.85) -> Tuple[bool, float]:
    """
    Check if transcription matches target word
    Returns (is_match, confidence)
    """
    similarity = calculate_similarity(target, transcription)
    is_correct = similarity >= threshold
    
    return is_correct, similarity

def find_best_match(target: str, candidates: list) -> Tuple[str, float]:
    """
    Find the best matching candidate from a list
    Returns (best_match, similarity_score)
    """
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        score = calculate_similarity(target, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match, best_score

