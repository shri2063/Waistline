k = "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"
from langchain.llms import OpenAI
from langchain import PromptTemplate
CATEGORY_LIST = ["I don't know","Category 1","Category 2","Category 3",
                 "Category 4","Category 5","Category 6","Category 7",
                 "Category 8","Category 9", "Category 10", "Category 11"]
CATEGORY_DICT = {
    0: "I don't know",
    1: "Involves neck or sleeves",
    2: "Chest size is smaller",
    3: "Chest size is larger",
    4: "Shoulder size is larger",
    5: "Shoulder size is smaller",
    6: "Waist size is larger",
    7: "Waist size is smaller",
    8: "tshirt length is long",
    9: "tshirt length is short",
    10: "tshirt in general undersize",
    11: "tshirt in general oversize"
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
multiple examples for each issue category. In your Answer just mention the Category name, for example - Category 1. Please note you have to select only from the given categories. Strictly do not improvise or innovate new  categories If the category  cannot be decided using the information provided answer
with "I don't know"

Context: 
Category 1 -  Involves neck or sleeves. Examples for this category: neck opening too tight, neck opening
 too wide, sleeves are too short, sleeves are too long, length of both sleeves do not match
Category 2- Chest size is smaller. Examples for this category: tshirt tight as chest, not fitting 
near chest,size smaller for chest, cannot breathe because too tight.  
Category 3- Chest size is larger. Examples for this category: tshirt loose at chest, loose fitting 
near chest,size larger for chest,tshirt wide near chest.
Category 4- Shoulder size is larger. Examples for this category: tshirt loose at shoulder, loose fitting 
near shoulder,shoulders are dropping down,tshirt wide near shoulder.
Category 5-  Shoulder size is smaller. Examples for this category: tshirt tight at shoulder, tight fitting 
near shoulder,size smaller for shoulder,tshirt underfitting near shoulder.
Category 6- Waist size is larger. Examples for this category: tshirt loose at stomach, loose fitting 
near waist,tshirt appears oversized at waist or stomach,tshirt wide near waist.
Category 7-  Waist size is smaller. Examples for this category: tshirt tight at stomach, stomach is clearly visible
,size smaller for waist,tshirt underfitting near waist or stomach.
Category 8- tshirt length is long. Examples for this category: tshirt too long, tshirt going below waist
tshirt length oversized,tshirt dropping down,tshirt wide in length.
Category 9-  tshirt length is short. Examples for this category: tshirt too short, tight not covering below waist 
,size smaller in length,tshirt underfitting in length.
Category 10-  tshirt in general undersize. Examples for this category: tshirt size is S but expected is M, tshirt size is M but expected is L, tshirt tight fitting 
,size smaller for tshirt,tshirt underfitting.
Category 11-  tshirt in general oversize. Examples for this category: tshirt size is L but expected is M, tshirt size is M but expected is S, tshirt loose fitting 
,size larger for tshirt,tshirt overfitting..

 


Question: {query}

Answer: """


#encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def generate_sizing_category_for_issue(query: str):
    prompt_template = PromptTemplate(input_variables=["query"], template=template_to_identify_sizing_category)
    response = openai_llm(prompt_template.format(query = query))
    print("Category response", str(response))
    for index, category in enumerate(CATEGORY_LIST):
        if category in response:
            return str(index)
    return str(0)