from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

tokenizer = MistralTokenizer.v1()


def count_tokens(text:str, tokenizer:MistralTokenizer=tokenizer):
    # Create a ChatCompletionRequest with the text
    request = ChatCompletionRequest(messages=[UserMessage(content=text)])
    # Use encode_chat_completion to get the tokens
    tokens = tokenizer.encode_chat_completion(request).tokens
    return len(tokens)


if __name__ == "__main__":
    # Test with an empty string
    empty_token_count = count_tokens("", tokenizer)
    print("Tokens for empty string:", empty_token_count)

    # Test with a non-empty string
    non_empty_token_count = count_tokens("Hello, world!", tokenizer)
    print("Tokens for 'Hello, world!':", non_empty_token_count)

    # Calculate the difference
    print("Difference:", non_empty_token_count - empty_token_count)