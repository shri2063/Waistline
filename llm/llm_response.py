
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
        "query": "The tshirt  is just bad",
        "answer": "I understand you are finding the tshirt  bad. Still I would like what is precise problem. Is it that pant has missing components or  quality of garment is thin or the tshirt dimensions are not correct"
    },
    {

        "query": "the tshirt has a hole of a size of coin near the sleeve",
        "answer": "Thanks . Please let me know If I have understood your concern correctly. The t-shirt you bought has quality issue because it has a hole inside it. Noting down the query:Issue:Quality:hole"
    },
    {
        "query": "Well. I am not sure if you understood correctly, but yes we can proceed with this understanding",
        "answer":"Great!  Give me a moment I will revert back"
    },
    {
        "query": "the tshirt  length is 41 cm while I expected it to 39 sm",
        "answer": "Thanks . Please let me know If I have understood your concern correctly. The tshirt you bought has sizing issue because its size is 41 cm when actual size should be 39 cm. Also may I know the size of tshirt."
    },
    {
        "query": "The size of tshirt is L size ",
        "answer":  "Thanks . Please let me know If I have understood your concern correctly. "
                    "The tshirt you bought has sizing issue because its size is 41 cm when actual size should be 39 cm. Noting down the query:"
                    "Issue:Sizing:Size:M: tshirt length is more than expected"
    },
    {
        "query": "I think you more or less got the point",
        "answer": "Great!  Give me a moment I will revert back"
    },
    {
        "query": "In the tshirt I bought one sleeve is longer than another",
        "answer": "I understand you are finding  in the tshirt one sleeve longer than another. "
                  "May I know the size of tshirt"
    },

    {
        "query": "The size of tshirt is S size ",
        "answer":  "Thanks . Please let me know If I have understood your concern correctly. "
                    "The tshirt you bought has sizing issue because length of both sleeves is different. Noting down the query:"
                    "Issue:Sizing:Size:M: mismatch in length of sleeves"
    },
    {
        "query": "In the tshirt I bought it feels tighter in the neck area. Tshirt size is L",
        "answer": "I understand you are finding that neck opening of tshirt is short."
                  "Please let me know If I have understood your concern correctly. Issue: Sizing: Size:L:short neck opening"
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


# now create the few shot prompt template
ai_introduction = "Greetings, I'm Rachel, and I'm delighted to welcome you to Waist Lyne. This serves as our MVP concept, showcasing how AI can assist " \
                  "buyers in addressing sizing or quality concerns, mirroring the support provided by a customer associate. At present, I'm equipped to " \
                  "recognize defects exclusively in T-shirts. Feel free to upload an image of a " \
                  "T-shirt with any defects or incorrect sizing, or alternatively, you can choose to download one from our catalog of T-shirt images. "

messages = [
    {'role': 'system', 'content': 'The following are exerpts from conversations with an AI assistant.  Company name is Waistlyne and your name is Rachel. This platform is basically MVP concept, showcasing how AI can assist buyers in addressing sizing or quality concerns, mirroring the  support provided by a customer associate. At present, '
                                  'AI is equipped to recognize defects exclusively in T-shirts. User is facing some problem in garments it has bought from the e-commerce store. AI is basically chat support of the '
                                  'e-commerce company who istrying to identify exact issue face by the problem'}]

def run_llm(query):
    result_string = prefix + ""
    for example in examples:
        result_string += f"Query: {example['query']}\nAnswer: {example['answer']}\n\n"


    list = encoding.encode(result_string)
    print(len(list))
    answer = openai_llm(get_revised_prompt_template().format(query=query))
    new_query = {"query": query, "answer": answer}
    examples.append(new_query)
    #print(examples)
    messages.append({'role': 'user', 'content': f"{query}"})
    messages.append({'role': 'assistant',
                     'content': f"{answer}"})

    return answer


def get_revised_prompt_template():
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n")


# print(openai(few_shot_prompt_template.format(query = query)))
# ----------------------------------------USING OPENAI DIRECTLY-----------------------------------------------------------


def chatgpt_call(prompt, model="gpt-3.5-turbo"):
    openai_llm.api_key = k
    response = openai_llm.ChatCompletion.create(
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
