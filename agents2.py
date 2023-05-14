from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
import os


def main():

    video_ideas = """1. A tutorial on how to use Anthropic's language model to analyze and find answers in podcast episodes.
        2. An interview with the creators of Assembly AI's Lemur framework, discussing its features and potential applications.
        3. A deep dive into the technology behind vector databases, exploring how they work and their impact on AI and machine learning.
        4. A case study on how a company or organization has successfully implemented a vector database for their data storage and retrieval needs.
        5. A comparison video of different vector databases, highlighting their strengths and weaknesses and helping viewers choose the best one for their needs.
        """
        
    

    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["SERPER_API_KEY"] = "d8c0597ad49019f1bbc962b915bddc32d96be5a8"
    llm = OpenAI(temperature=0.3)
    tools1 = load_tools(["google-serper"], llm=llm)
    agent1 = initialize_agent(tools1, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent2 = initialize_agent(tools1, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    agent1.agent.llm_chain.prompt.template = """You are a Strategist at the company AI SynpaseXIT. You are an expert at Digital Youtube content strategy.
    Your task is to make data-driven decisions to imporve the engagement of our videos.\n """ + agent1.agent.llm_chain.prompt.template 

    agent2.agent.llm_chain.prompt.template = """You are a Researcher at the company AI SynpaseXIT. You are an expert at Researching Youtube content on the internet.
    Your task is to find upto date data on the internet based on inputs provided to imporve the engagement of our videos.\n """ + agent2.agent.llm_chain.prompt.template 

    print(agent1.run("Based on the following inputs:" + video_ideas + ".\nGenerate a new original idea not present in the inputs that we can use to create new video content which will make Youtube videos more engaging for viewers.").Final Answer)

"""
    new_ideas = agent1.brainstorm_ideas()
    agent2.search_internet(new_ideas)

    best_ideas = agent2.prioritize_ideas(new_ideas)
    print("Top 5 Video Ideas:")
    for idea in best_ideas:
        print(idea)"""

if __name__ == "__main__":
    main()