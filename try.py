import transformers
pipeline = transformers.pipeline("text-generation", model="pfnet/plamo-2-1b", trust_remote_code=True)
print(pipeline("The future of artificial intelligence technology is ", max_new_tokens=32))
