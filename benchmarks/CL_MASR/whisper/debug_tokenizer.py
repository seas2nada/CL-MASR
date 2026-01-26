
from model import ProgressiveWhisperTokenizer as WhisperTokenizer
try:
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
    print(f"Has _additional_special_tokens: {hasattr(tokenizer, '_additional_special_tokens')}")
    print(f"additional_special_tokens: {tokenizer.additional_special_tokens}")
    
    new_tokens = ["<|new_lang|>"]
    # Try the failing line
    try:
        tokenizer.additional_special_tokens += new_tokens
        print("Successfully added to _additional_special_tokens")
    except AttributeError as e:
        print(f"Failed to add to _additional_special_tokens: {e}")

    # Try correct way
    print("Trying add_tokens with special_tokens=True")
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    print(f"Num added: {num_added}")
    print(f"additional_special_tokens after add: {tokenizer.additional_special_tokens}")
    
except Exception as e:
    print(f"An error occurred: {e}")
