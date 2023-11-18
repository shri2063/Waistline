k = "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"
from langchain.llms import OpenAI
from langchain import PromptTemplate
template = """Answer the question based on the context below. You have to select correct Issue Category for 
the User query provided in the Input query. In the context below you have been provided multiple Issue categories along with 
multiple examples for each issue category. In your Answer just mention the issue name, for example - issue: Chest size is smaller
or issue: tshirt in general oversize.If the question cannot be answered using the information provided answer
with "I don't know"

Context: 
Category 1- Chest size is smaller. Examples for this category: tshirt tight as chest, not fitting 
near chest,size smaller for chest, cannot breathe because too tight.  
Category 2- Chest size is larger. Examples for this category: tshirt loose at chest, loose fitting 
near chest,size larger for chest,tshirt wide near chest.
Category 3- Shoulder size is larger. Examples for this category: tshirt loose at shoulder, loose fitting 
near shoulder,shoulders are dropping down,tshirt wide near shoulder.
Category 4-  Shoulder size is smaller. Examples for this category: tshirt tight at shoulder, tight fitting 
near shoulder,size smaller for shoulder,tshirt underfitting near shoulder.
Category 5- Waist size is larger. Examples for this category: tshirt loose at stomach, loose fitting 
near waist,tshirt appears oversized at waist or stomach,tshirt wide near waist.
Category 6-  Waist size is smaller. Examples for this category: tshirt tight at stomach, stomach is clearly visible
,size smaller for waist,tshirt underfitting near waist or stomach.
Category 7- tshirt length is long. Examples for this category: tshirt too long, tshirt going below waist
tshirt length oversized,tshirt dropping down,tshirt wide in length.
Category 8-  tshirt length is short. Examples for this category: tshirt too short, tight not covering below waist 
,size smaller in length,tshirt underfitting in length.
Category 9-  tshirt in general undersize. Examples for this category: tshirt size should be L but it is M, tshirt tight fitting 
,size smaller for tshirt,tshirt underfitting.
Category 10-  tshirt in general oversize. Examples for this category: tshirt size should be M but it is L, tshirt loose fitting 
,size larger for tshirt,tshirt overfitting..
Category 11-  Involves neck or sleeves. Examples for this category: neck opening too tight, neck opening
 too wide, slleves are too short, sleeves are too long, length of both sleeves do not match
 


Question: {query}

Answer: """

openai_llm = OpenAI(
    model_name="text-davinci-003",
    openai_api_key=k
)
#encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def find_sizing_category_for_issue(query: str):
    prompt_template = PromptTemplate(input_variables=["query"], template=template)
    response = openai_llm(prompt_template.format(query = query))
    return response