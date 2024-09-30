import os
#from groq import Groq
from openai import OpenAI

#LLM model - we use Meta's open source LLama model.
MODEL = "llama3-8b-8192"
# Controls randomness: lowering results in less random completions.
# As the temperature approaches zero, the model will become deterministic
# and repetitive.
temperature=0.7

# The maximum number of tokens to generate. Requests can use up to
# 32,768 tokens shared between prompt and completion.
max_tokens=1024

# Controls diversity via nucleus sampling: 0.5 means half of all
# likelihood-weighted options are considered.
top_p=1

# A stop sequence is a predefined or user-specified text string that
# signals an AI to stop generating content, ensuring its responses
# remain focused and concise. Examples include punctuation marks and
# markers like "[end]".
stop=None

# If set, partial message deltas will be sent.
stream=False


#retrieve api GROQ key
#API_KEY = os.environ.get("GROQ_API_KEY")
#dummy API key for openai client using local ollama model
API_KEY = "ollama"
print(f"{API_KEY}")
# client = Groq(
#     api_key=API_KEY
#     )

#initiate local client using openai API
client = OpenAI(
    base_url = "http://localhost:11434/v1", 
    api_key = API_KEY
)

# Function to access 
def retrieve_message_from_ollama(messages_list : list, model):

    #access the openAI chat completions API
    chat_completion = client.chat.completions.create (
        model=model, 
        messages=messages_list,
        temperature=temperature, 
        max_tokens=max_tokens, 
        top_p=top_p
    )

    #retrieve the response from LLM
    return chat_completion.choices[0].message.content


# Function to access 
def retrieve_message_from_groq(messages_list : list, model):

    #access the Groq chat completions API
    chat_completion = client.chat.completions.create (
        model=model, 
        messages=messages_list,
        temperature=temperature, 
        max_tokens=max_tokens, 
        top_p=top_p
    )

    #retrieve the response from LLM
    return chat_completion.choices[0].message.content