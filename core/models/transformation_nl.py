import re

# Enhanced visually similar Devanagari character groups
similar_char_groups = {
    'ा': ['र'],  # aa ki matra vs ra - but NOT puranviram
    'र': ['ा', 'य'],
    'ल': ['य', 'व', 'न'],
    'य': ['ल', 'प', 'र', 'ग', 'ज'],  # Added ज for जग/यग case
    'प': ['य', 'घ', 'क', 'र'],
    'घ': ['प', 'ध'],
    'ध': ['घ'],
    'म': ['थ'],
    'थ': ['म'],
    'क': ['फ', 'प', 'र'],
    'फ': ['क'],
    'ग': ['य'],
    'в': ['ल'],
    'न': ['ब', 'ल'],
    'ब': ['न'],  # Added अ for अब/जब case
    'ठ': ['भ'],
    'भ': ['ठ'],
    'ट': ['त'],
    'त': ['ट'],
    'श': ['स'],
    'स': ['श'],
    'ज': ['य', 'अ'],  # Added for जग/यग and अब/जब cases
    'द': ['ढ'],
    'ढ': ['द'],
    'अ': ['ज'],  # For अब/जब confusion
    'ञ': ['स'],  # For यज्ञ/यस case - ञ can be confused with स
    'ो': ['ा']  # o ki matra can be confused with aa ki matra
}

# Dictionary for specific multi-character similarities (e.g., char vs char+matra)
multi_char_similarities = {
    'क': ['पा'],
    'पा': ['क'],
    'रा' : ['श'],
    'श' : ['रा'],

}

def build_similarity_lookup(groups):
    lookup = {}
    for key, vals in groups.items():
        group = [key] + vals
        for char in group:
            if char not in lookup:
                lookup[char] = []
            lookup[char].extend([c for c in group if c != char and c not in lookup[char]])
    return lookup

devanagari_similarity_lookup = build_similarity_lookup(similar_char_groups)

def clean_word(word):
    """Remove dots, poorna virams, and normalize spaces."""
    if not word or word.lower() in ['none', 'null', 'nan']:
        return ''
    return re.sub(r'[।.,\-]', '', str(word)).strip()

def has_punctuation_difference(word1, word2):
    """Check if words differ only by punctuation."""
    clean1 = clean_word(word1)
    clean2 = clean_word(word2)
    return clean1 == clean2 and word1 != word2

def group_devanagari_chars(word):
    """Groups consonants with their following matras into syllabic units."""
    # List of Devanagari vowel signs (matras) and other combining marks
    matras = "ािीुूृॄॅेैॉोौ्" 
    
    if not word:
        return []
    
    units = []
    current_unit = word[0]
    
    for i in range(1, len(word)):
        char = word[i]
        if char in matras:
            # If it's a matra, attach it to the current unit
            current_unit += char
        else:
            # Otherwise, the last unit is complete. Start a new one.
            units.append(current_unit)
            current_unit = char
            
    units.append(current_unit) # Add the last unit
    return units

def is_visually_similar(word1, word2, lookup=devanagari_similarity_lookup):
    """Upgraded visual similarity check that compares syllabic units."""
    if not word1 or not word2:
        return False
    if word1 == word2:
        return True

    # Group both words into syllabic units
    units1 = group_devanagari_chars(word1)
    units2 = group_devanagari_chars(word2)

    # 1. Check for a single unit substitution
    if len(units1) == len(units2):
        diff_units = [(u1, u2) for u1, u2 in zip(units1, units2) if u1 != u2]
        
        if len(diff_units) == 1:
            ocr_unit, ref_unit = diff_units[0]
            
            # Check for similarity using both single-char and multi-char dictionaries
            if (ocr_unit in lookup and ref_unit in lookup[ocr_unit]) or \
               (ocr_unit in multi_char_similarities and ref_unit in multi_char_similarities[ocr_unit]):
                return True
            
    return False

def check_merged_words(transcribed, reference_words):
    """Check if a transcribed word is actually multiple reference words."""
    if not transcribed or not reference_words:
        return None
    
    joined_ref = ''.join(reference_words)
    if transcribed == joined_ref:
        return reference_words
    
    for i in range(len(reference_words)-1):
        merged = ''.join(reference_words[i:i+2])
        if transcribed == merged:
            return reference_words[i:i+2]
    
    return None

def transform_word_evaluations_nl(word_evaluations):
    """Enhanced transformation function using syllabic unit comparison."""
    
    # First pass: clean all words and handle None values
    for w in word_evaluations:
        ref_orig = w.get('reference_word', '')
        trans_orig = w.get('transcribed_word', '')
        
        ref = clean_word(ref_orig)
        trans_clean = clean_word(trans_orig)

        if not trans_clean and ref:
            w['reason_diff'] = (w.get('reason_diff', '') + 
                               f" [Missing word: expected '{ref}']").strip()
            w['match'] = False
            continue

        if has_punctuation_difference(trans_orig, ref_orig):
            w['reason_diff'] = (w.get('reason_diff', '') +
                               f" [Punctuation difference: '{trans_orig}'→'{ref_orig}']").strip()
            w['transcribed_word'] = ref_orig
            w['match'] = True
            continue
        
        if trans_orig != trans_clean:
            w['reason_diff'] = (w.get('reason_diff', '') +
                               f" [Cleaned punctuation: '{trans_orig}'→'{trans_clean}']").strip()
            w['transcribed_word'] = trans_clean

    # Second pass: check for merged words and visual similarity
    all_refs = [clean_word(w.get('reference_word', '')) for w in word_evaluations]
    
    for i, w in enumerate(word_evaluations):
        ref = clean_word(w.get('reference_word', ''))
        trans_clean = clean_word(w.get('transcribed_word', ''))
        
        if not ref or not trans_clean or w.get('match', False):
            continue

        merged = check_merged_words(trans_clean, all_refs[max(0, i-1):min(len(all_refs), i+3)])
        if merged:
            w['reason_diff'] = (w.get('reason_diff', '') +
                               f" [Detected merged words: '{trans_clean}' split into '{' '.join(merged)}']").strip()
            w['transcribed_word'] = merged[0]
            w['match'] = True
            continue

        if ref != trans_clean:
            if is_visually_similar(trans_clean, ref):
                w['reason_diff'] = (w.get('reason_diff', '') +
                                   f" [Corrected by visual similarity: '{trans_clean}'→'{ref}']").strip()
                w['transcribed_word'] = ref
                w['match'] = True

    return word_evaluations