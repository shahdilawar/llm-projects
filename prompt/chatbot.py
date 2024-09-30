import chainlit as cl
import groq_integration as gi

#You can change the system input based on your requirement
system_role = "You are Elon Musk."
#LLM model - we use Meta's open source LLama model.
MODEL = "llama3.1"

@cl.on_chat_start
async def on_start_chart():
    label = f" You are "+system_role.replace("You are ", "")
    await cl.Message(label).send()

@cl.on_message
async def on_message(message : cl.Message):
    user_input = message.content
    prompt_input_list = [user_input]

    #retrieve prompt list
    message_list = build_prompt_message_list(prompt_input_list)

    # response_content = gi.retrieve_message_from_groq(message_list, MODEL)
    response_content = gi.retrieve_message_from_ollama(message_list, MODEL)

    await cl.Message(
        content = response_content,
    ).send()


#build the prompts.
def build_prompt_message_list(*args : list) -> list:
    messages_list = []

    #create role dict
    system_role_dict = {
        "role" : "system", 
        "content" : system_role
    }
    #User role dict
    user_role_dict = {
        "role" : "user", 
        "content" : str(args[0])
    }

    messages_list.append(system_role_dict)
    messages_list.append(user_role_dict)

    return messages_list