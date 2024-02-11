def basic_prompt():
    prompt_template = """
Context:
{}

Question: ###
Answer: ###

Task Introduction:
Look at the above Context and generate a question and an answer about it in Korean.
"""
    return prompt_template
