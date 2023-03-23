# c-tokenizer
GPT Tokenizers for C#

These two classes can be use to tokenize (and count) tokens for OpenAI's GPT3, GPT4, and GPT 3.5 Turbo Models

The cl100k class also counts tokens for the ADA 002 embedding model

This code has been extracted from a larger project. 

You will need to change the hard coded path to the tiktoken files.

More details about how tokens work can be found here : https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

For GPT3 models like text-davinici-003, use pk50 as the tokenizer

For GPT 3.5, GPT 4, and text-embedding-ada-002, use bl100k as the tokenizer
