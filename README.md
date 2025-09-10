# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date: 10/09/2025
# Register no. 212223060289
# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools
# Scenario: Product Feature Comparison Across AI Tools

# AI Tools Required:

1.OpenAI GPT (ChatGPT API)

2.Hugging Face Transformers

3.LangChain Framework (optional)

4.Google Generative AI API (optional)

# Explanation:

Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that. 

Steps followed:

1.Define the persona: Data Analyst writing Python code to analyze customer feedback.

2.Use Hugging Face for quick sentiment analysis.

3.Use OpenAI GPT for deeper insights (sentiment, keywords, suggestions).

4.Compare and analyze outputs for consistency and actionability.


# Conclusion:

# -------------------------------
# Import Required Libraries
# -------------------------------
from transformers import pipeline
import openai
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Google Generative AI (optional)
```
try:
    import google.generativeai as genai
except ImportError:
    genai = None
```

# -------------------------------
# Input Text (Two Customer Reviews)
# -------------------------------
review_product_A = """
The laptop has excellent performance and battery life.
However, the keyboard feels cheap and the speakers are below average.
"""

review_product_B = """
This laptop looks premium and the display quality is outstanding.
But the battery drains quickly and the fan makes too much noise.
"""


# -------------------------------
# 1. Hugging Face Transformers
# -------------------------------
```
print("=== Hugging Face Sentiment Analysis for Both Products ===")

sentiment_analyzer = pipeline("sentiment-analysis")

hf_sentiment_A = sentiment_analyzer(review_product_A)
hf_sentiment_B = sentiment_analyzer(review_product_B)

print("Product A Sentiment:", hf_sentiment_A[0])
print("Product B Sentiment:", hf_sentiment_B[0])

```
# -------------------------------
# 2. OpenAI GPT (ChatGPT API)
# -------------------------------
```
print("\n=== OpenAI GPT Feature Comparison ===")

openai.api_key = "YOUR_OPENAI_API_KEY"

prompt = f"""
```
Compare the following two customer reviews:

Product A Review: {review_product_A}

Product B Review: {review_product_B}

Tasks:
1. Extract pros and cons of each product.
2. Identify the overall sentiment (positive/negative/neutral).
3. Suggest which product seems better overall and why.
"""
```
gpt_response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=250,
    temperature=0.5
)

gpt_output = gpt_response["choices"][0]["text"].strip()
print(gpt_output)

```
# -------------------------------
# 3. LangChain Framework
# -------------------------------
```
print("\n=== LangChain Structured Analysis ===")

template = """
You are a market research analyst. Compare two products based on customer reviews.

Product A Review: {review_A}
Product B Review: {review_B}

Provide structured output:
- Product A: Pros / Cons
- Product B: Pros / Cons
- Final Recommendation
"""
prompt_template = PromptTemplate(
    input_variables=["review_A", "review_B"], 
    template=template
)

llm = LangChainOpenAI(openai_api_key="YOUR_OPENAI_API_KEY", model_name="text-davinci-003")
chain = LLMChain(llm=llm, prompt=prompt_template)

langchain_result = chain.run(review_A=review_product_A, review_B=review_product_B)
print(langchain_result)
```

# -------------------------------
# 4. Google Generative AI (Optional)
# -------------------------------
```
if genai:
    print("\n=== Google Generative AI (Gemini) ===")
    genai.configure(api_key="YOUR_GOOGLE_API_KEY")

    model = genai.GenerativeModel("gemini-pro")
    gemini_prompt = f"""
    Compare the following two reviews:
    
    Product A: {review_product_A}
    Product B: {review_product_B}

    Summarize pros, cons, and give a recommendation.
    """

    response = model.generate_content(gemini_prompt)
    print(response.text)
else:
    print("\n[Google Generative AI not installed. Skipping this step.]")
```
## How the AI tools works:

1.Hugging Face → gives sentiment for each product.

2.OpenAI GPT → extracts pros/cons + final recommendation.

3.LangChain → enforces structured output format.

4.Google Generative AI → (optional) provides another independent comparison

# Result: 

 The corresponding Python code was executed successfully, proving that sentiment analysis and keyword extraction can be enhanced by combining multiple AI tools.
