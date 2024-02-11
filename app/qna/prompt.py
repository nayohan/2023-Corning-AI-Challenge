from langchain.prompts import PromptTemplate

def basic_prompt():
    # Define our prompt
    prompt_template = """You will be shown dialogues between Speaker 1 and Speaker 2. Please read and understand given Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n```
    ```
    Context:
    {context}
    ```
    
    ```
    Dialogue Session:
    {question}
    ```
    Task Introduction:
    After reading the Dialogue Session, please create an appropriate response in the parts marked ###.
    ```
    
    Task Result:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    return prompt


    # Speaker 1: {question}
    # Speaker 2: ###