from typing import Tuple

from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile
# from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
# from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
# from third_parties.twitter import scrape_user_tweets
from output_parsers import summary_parser, Summary


def ice_break_with(name: str) -> Tuple[Summary, str]:
    # linkedin_username = linkedin_lookup_agent(name=name)
    # linkedin_data = scrape_linkedin_profile(
    #     linkedin_profile_url=linkedin_username, mock=True
    # )

    # twitter_username = twitter_lookup_agent(name=name)
    # tweets = scrape_user_tweets(username=twitter_username, mock=True)

    information="""
      Elon Reeve Musk, born June 28, 1971, is a businessman and investor known for his key roles in the space company SpaceX and the automotive company Tesla, Inc. Other involvements include ownership of X Corp., the company that operates the social media platform X (formerly known as Twitter), and his role in the founding of The Boring Company, xAI, Neuralink, and OpenAI. He is one of the wealthiest individuals in the world; as of August 2024 Forbes estimates his net worth to be US$247 billion.[3]
      Musk was born in Pretoria to Maye (née Haldeman), a model, and Errol Musk, a businessman and engineer. Musk briefly attended the University of Pretoria before immigrating to Canada at the age of 18, acquiring citizenship through his Canadian-born mother. Two years later he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but dropped out after two days and, with his brother Kimbal, co-founded the online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In October 2002, eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002.
    """



    # summary_template = """
    # given the information about a person from linkedin {information},
    # and their latest twitter posts {twitter_posts} I want you to create:
    # 1. A short summary
    # 2. two interesting facts about them 

    # Use both information from twitter and Linkedin
    # \n{format_instructions}
    # """

    summary_template = """
      given the information about a person from {information},
      I want you to create:
      1. A short summary
      2. two interesting facts about them 

      \n{format_instructions}
      """

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm = ChatOllama(model="llama3.1", temperature=0) # Model should be installed locally

    chain = summary_prompt_template | llm | summary_parser

    res = chain.invoke(input={ "information": information })

    # profile_pic_url = linkedin_data.get("profile_pic_url")
    profile_pic_url = "https://fastly.picsum.photos/id/913/200/300.jpg?hmac=DjpzGA27POHBn03vW7UxM5gI9phMxuAZ4hSKcRfJD9Y"

    return res, profile_pic_url


if __name__ == "__main__":
    load_dotenv()

    print("Ice Breaker Enter")
    data = ice_break_with(name="Elon Musk")
    print("data: ")
    print(data)
