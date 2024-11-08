import openai

k = "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"
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
import  tiktoken


# ------------------------------------- USING LANGCHAIN CONVERSATION CHAIN -----------------------------------------------


llm = OpenAI(openai_api_key=k)
memory = ConversationBufferMemory()
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."
memory_2 = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory_2.save_context({"input": "Hello"}, {"output": "Hi , how can I help you today"})
memory_2.save_context({"input": "I have received a damage tshirt"}, {
    "output": "I see. Can you be more specific. Is there hole or stain mark in the tshirt or length of particular dimension is incorrect"})
memory_2.save_context({"input": "There is a hole near the sleeves"},
                      {"output": "I understood the problem. let me see what I can do"})

conversation = ConversationChain(
    llm=llm,
    memory=memory_2,
    verbose=True
)
# print(conversation.clean_tshirts(input = "What would be good time to take coffee break"))

# -------------------------USING LANGCHAIN PROMPT TEMPLATE--------------------------------
template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: {query}

Answer: """

openai_llm = OpenAI(
    model_name="text-davinci-003",
    openai_api_key=k
)
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
prompt_template = PromptTemplate(input_variables=["query"], template=template)

# print(openai(prompt_template.format(query = "Which libraries and model providers offer LLMs?")))


# -------------------------USING LANGCHAIN FEW SHORT  PROMPT TEMPLATE--------------------------------
# USING OPENAI LIBRARY WITHOUT LANGCHAIN FRAMEWORK
prompt = """The following are exerpts from conversations with an AI
assistant. The assistant is always sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 

User: How are you?
AI: I can't complain but sometimes I still do.

User: What time is it?
AI: It's time to get a watch.

User: What is the meaning of life?
AI: """

# print(openai(prompt))


# USING LANGCHAIN FRAMEWORK

# create our examples
examples = [
    {
        "query": "Hi",
        "answer": "Hi.Thanks for reaching out how can I help you out today."
    },
    {
        "query": "I just got a tshirt  from your site and it seems that it has two of the buttons in the top missing",
        "answer": "I understand you are finding missing buttons from tshirt you bought from our site . However, currently I can help in assisting queries only related to sizing and quality . I am really sorry and hopefully soon I can "
                  "help in solving problem related to missing items too "
    },
    {
        "query": "Tshirt I bought from your site showed different colour on the website and the one I bought has different colour",
        "answer": "I understand you are finding colour in the tshirt not as per your expectation . However, currently I can help in assisting queries only related to sizing and quality . I am really sorry and hopefully soon I can "
                  "help in solving problem related to colour mismatch items too "
    },
    {
        "query": "The white coloured dress is not good",
        "answer": "I understand you are finding  white coloured dress you bought from our site not good. However, currently I can help in assisting queries only related to tshirts. I am really sorry and hopefully soon I can "
                  "help in solving problem related to dress too "
    },
    {
        "query": "There is a problem in the tshirt I bought from your e-commerce site",
        "answer": "I understand you are facing problem in the t-shirt you bought from our site. Can you explain more about the problem you are facing. Is it defect like stain or hole or missing buttons "
    },
    {
        "query": "the tshirt  length is incorrect",
        "answer": "Thanks . I understand that you find tshirt length incoorect, but can you exaplain in detail if tshirt length is short or long"
    },
    {
        "query": "the tshirt  length is 41 cm while I expected it to 39 sm",
        "answer": "Thanks . Please let me know If I have understood your concern correctly. The tshirt you bought has sizing issue because its size is 41 cm when actual size should be 39 cm. Noting down Issue:Sizing:tshirt length is more than expected"
    },

    {
        "query": "The tshirt  is just bad",
        "answer": "I understand you are finding the tshirt  bad. Still I would like what is precise problem. Is it that pant has missing components or  quality of garment is thin or the tshirt dimensions are not correct"
    },
    {
        "query": "the tshirt  size is incorrect",
        "answer": "Thanks . I understand that you find tshirt size incoorect, but can you exaplain in detail if tshirt size is smaller or larger than you expected"
    },
    {
        "query": "I ordered tshirt of size M but received tshirt of Size L",
        "answer": "I understand you are finding that neck opening of tshirt is short. Noting down the Issue: Sizing:tshirt is oversized"
    },
    {
        "query": "I ordered tshirt of size L but received tshirt of Size S",
        "answer": "I understand you are finding that neck opening of tshirt is short. Noting down the Issue: Sizing:tshirt is undersized"
    },
    {

        "query": "the tshirt has a hole of a size of coin near the sleeve",
        "answer": "Thanks . Please let me know If I have understood your concern correctly. The t-shirt you bought has quality issue because it has a hole inside it. Noting down the Issue:Quality:hole"
    },
    {

        "query": "the tshirt has a yellow stain near the chest",
        "answer": "Thanks . The t-shirt you bought has quality issue because it has a yell. Noting down the Issue:Quality:stain"
    },

    {
        "query": "In the tshirt I bought one sleeve is longer than another",
        "answer": "I understand you are finding  in the tshirt one sleeve longer than another. Please let me know If I have understood your concern correctly."
                  "Noting down the Issue:Sizing: mismatch in length of sleeves"
    },



]


# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are exerpts from conversations with an AI assistant.  Company name is Waistlyne and your name is Rachel. This platform is basically MVP concept, showcasing how AI can assist buyers
 in addressing sizing or quality concerns, mirroring the  support provided by a customer associate. At present, AI is equipped to recognize defects exclusively 
 in T-shirts.Whenever the issue deals with length or size smaller/larger it comes into Sizing issue and when there is a hole or stain  or any poor quality it comes into Quality issue. You need to identify wehter it is sizing issue or Quality issue.
  . User is facing some problem in garments it has bought from the e-commerce store. AI is basically chat support of the e-commerce company who is
trying to identify exact issue face by the problem .Once you have identified the issue do not end the conversation until you note down the query:Issue.... and ensure that you  receive acknowledgement from  the user that query is correct. Also, note this is an experimental setup, so if user does not have a tshirt, guide him
 to download one from catalogue present in the left sidebar. .  Here are some examples just for reference """
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

secret_key = "Please enter your secret key"
# now create the few shot prompt template
ai_introduction = ("Hi there! This platform serves as our MVP concept, demonstrating how AI can "
                   "aid buyers in assessing the fit of garments. I'd appreciate it if you "
                   "could provide an image of your favorite t-shirt to help me understand "
                   "your preferred fit. Afterwards, you can upload any t-shirt, and I'll be "
                   "able to suggest how it might fit you")

            #"Presently, I can determine your ideal fitting measurements based on a snapshot of your favorite T-shirt.") \
            #" At this point, my capability is limited to sizing assessments for T-shirts, and it does not extend " \
            #"to other apparel styles. However, in the near future, I will be equipped to derive body measurements directly from your " \
            #"full-length image. Stay tuned for upcoming enhancements!"








messages = [
    {'role': 'system', 'content': 'The following are exerpts from conversations with an AI assistant.  Company name is Waistlyne and your name is Rachel. This platform is basically MVP concept, showcasing how AI can assist buyers in addressing sizing or quality concerns, mirroring the  support provided by a customer associate. At present, '
                                  'AI is equipped to recognize defects exclusively in T-shirts. User is facing some problem in garments it has bought from the e-commerce store. AI is basically chat support of the '
                                  'e-commerce company who is trying to identify exact issue face by the problem'}]

def run_llm(query, conversation_list):
    result_string = prefix + ""
    customized_examples = []
    for example in examples:
        result_string += f"Query: {example['query']}\nAnswer: {example['answer']}\n\n"
        customized_examples.append(example)
    if conversation_list is not  None:
        for index, conversation in enumerate(conversation_list):
            if index == len(conversation) - 1:
                result_string += f"Query: {conversation['query']}\nAnswer: {conversation['answer']}\n\n"
                customized_examples.append(conversation)


    list = encoding.encode(result_string)
    print(len(list))

    answer = openai_llm(get_revised_prompt_template(customized_examples).format(query=query))
    #print(examples)
    messages.append({'role': 'user', 'content': f"{query}"})
    messages.append({'role': 'assistant',
                     'content': f"{answer}"})

    return answer


def get_revised_prompt_template(conversation):
    return FewShotPromptTemplate(
        examples=conversation,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n")


# print(openai(few_shot_prompt_template.format(query = query)))
# ----------------------------------------USING OPENAI DIRECTLY-----------------------------------------------------------


def chatgpt_call(prompt, model="text-davinci-003"):
    openai.api_key = k
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]


def chatgpt_call_with_memory(query, model="gpt-3.5-turbo"):
    openai_llm.api_key = k
    response = openai_llm.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message["content"]




def chatgpt_conversation(prompt):
    messages.append({'role': 'user', 'content': f"{prompt}"})
    response = chatgpt_call_with_memory(messages)
    messages.append({'role': 'assistant',
                     'content': f"{response}"})
    return response

# prompt = "My name is Shrikant"
# response = chatgpt_call_with_memory(messages)
# chatgpt_conversation("Hello, my name is Andrea")
# chatgpt_conversation("Do you remember my name?")
# print(messages)
