def generate_variants(answer_text: str) -> list[str]:
    """
    Generate semantic variants for an answer string.
    """
    variants = []
    base = str(answer_text)
    
    # Raw
    variants.append(base)
    # Leading space
    variants.append(" " + base)
    
    # Lowercase
    if base.islower() is False:
        variants.append(base.lower())
        variants.append(" " + base.lower())
        
    # Titlecase
    if base.istitle() is False and base.islower():
        variants.append(base.title())
        variants.append(" " + base.title())
        
    # Uppercase
    if base.isupper() is False and len(base) <= 3: # Keep acronyms fully upper
        variants.append(base.upper())
        variants.append(" " + base.upper())
        
    return list(set(variants))

def filter_single_token_variants(tokenizer, variants: list[str]) -> list[str]:
    """
    Return only variants that encode to exactly 1 token under the given tokenizer.
    """
    valid = []
    for var in variants:
        # Avoid special tokens added by tokenizer
        tokens = tokenizer.encode(var, add_special_tokens=False)
        if len(tokens) == 1:
            valid.append(var)
    return valid
