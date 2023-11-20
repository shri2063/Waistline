k = "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"
from langchain.llms import OpenAI
from langchain import PromptTemplate
CATEGORY_LIST = ["I don't know","Category B","Category C","Category D",
                 "Category E","Category F","Category G","Category H",
                 "Category I","Category J", "Category K", "Category L"]
CATEGORY_DICT = {
    "A": "I don't know",
    "B": "Involves neck or sleeves",
    "C": "Chest size is smaller",
    "D": "Chest size is larger",
    "E": "Shoulder size is larger",
    "F": "Shoulder size is smaller",
    "G": "Waist size is larger",
    "H": "Waist size is smaller",
    "I": "tshirt length is long",
    "J": "tshirt length is short",
    "K": "tshirt in general undersize",
    "L": "tshirt in general oversize"
}
openai_llm = OpenAI(
    model_name="text-davinci-003",
    openai_api_key=k
)


template_to_respond_to_sizing_query = """You are AI bot helping to resolve customer query regarding sizing issue
in the tshirt. You have information about the tshirt available that is provided in the Context.Please answer
the query of the customer based upon the Context provided 

Context: {context}

Question : {query}

Answer : """

def generate_response_based_upon_sizing_calculations(query: str, context: str):
    prompt_template = PromptTemplate(input_variables=["query", "context"], template=template_to_respond_to_sizing_query)
    response = openai_llm(prompt_template.format(query = query,context = context))
    print("Sizing response", str(response))
    return response


template_to_identify_sizing_category = """Answer the question based on the context below. You have to select correct Issue Category for 
the User query provided in the Input query. In the context below you have been provided multiple Issue categories along with 
multiple examples for each issue category. In your Answer Strictly just mention a single letter which denotes the category. for example: J . Please note you have to select only from the given categories. Strictly do not improvise or innovate new  categories If the category  cannot be decided using the information provided answer
with "I don't know"

Context: 
Category A -  Involves neck or sleeves. Examples for this category: neck opening too tight, neck opening
 too wide, sleeves are too short, sleeves are too long, length of both sleeves do not match
Category B- Chest size is smaller. Examples for this category: tshirt tight as chest, not fitting 
near chest,size smaller for chest, cannot breathe because too tight.  
Category C- Chest size is larger. Examples for this category: tshirt loose at chest, loose fitting 
near chest,size larger for chest,tshirt wide near chest.
Category D- Shoulder size is larger. Examples for this category: tshirt loose at shoulder, loose fitting 
near shoulder,shoulders are dropping down,tshirt wide near shoulder.
Category E-  Shoulder size is smaller. Examples for this category: tshirt tight at shoulder, tight fitting 
near shoulder,size smaller for shoulder,tshirt underfitting near shoulder.
Category F- Waist size is larger. Examples for this category: tshirt loose at stomach, loose fitting 
near waist,tshirt appears oversized at waist or stomach,tshirt wide near waist.
Category G-  Waist size is smaller. Examples for this category: tshirt tight at stomach, stomach is clearly visible
,size smaller for waist,tshirt underfitting near waist or stomach.
Category H- tshirt length is long. Examples for this category: tshirt too long, tshirt going below waist
tshirt length oversized,tshirt dropping down,tshirt wide in length.
Category I-  tshirt length is short. Examples for this category: tshirt too short, tight not covering below waist 
,size smaller in length,tshirt underfitting in length.
Category J-  tshirt in general undersize. Examples for this category:  sizing is undersized, size is smaller than expected, tshirt size is S but expected is M, tshirt size is M but expected is L, tshirt tight fitting 
,size smaller for tshirt,tshirt underfitting.
Category K-  tshirt in general oversize. Examples for this category: tshirt size is larger than expected,tshirt size is L but expected is M, tshirt size is M but expected is S, tshirt loose fitting 
,size larger for tshirt,tshirt overfitting..

 


Question: {query}

Answer: """


#encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def generate_sizing_category_for_issue(query: str):
    prompt_template = PromptTemplate(input_variables=["query"], template=template_to_identify_sizing_category)
    response = openai_llm(prompt_template.format(query = query))
    print("Category response", str(response))
    for index, category in enumerate(CATEGORY_LIST):
        if response in category:
            print(CATEGORY_LIST[index])
            return CATEGORY_LIST[index]
    return CATEGORY_LIST[0]







    return CATEGORY_LIST[0]