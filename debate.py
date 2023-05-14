from typing import List, Dict, Callable
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)


from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "d8c0597ad49019f1bbc962b915bddc32d96be5a8"
class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()
        
    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function
        
    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message

class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
        tool_names: List[str],
        **tool_kwargs,
    ) -> None:
        super().__init__(name, system_message, model)
        self.tools = load_tools(tool_names, **tool_kwargs)

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        agent_chain = initialize_agent(
            self.tools, 
            self.model, 
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
            verbose=True, 
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )
        message = AIMessage(content=agent_chain.run(
            input="\n".join([
                self.system_message.content] + \
                self.message_history + \
                [self.prefix])))
        
        return message.content


names = {
    'Youtube Content Strategist': [
        'wikipedia',
        "google-serper"
    ],
    'Youtube Content Researcher': [
        'wikipedia',
        'google-serper'
    ],
}

video_ideas = """

1. A tutorial on how to use Anthropic's language model to analyze and find answers in podcast episodes.
2. An interview with the creators of Assembly AI's Lemur framework, discussing its features and potential applications.
3. A deep dive into the technology behind vector databases, exploring how they work and their impact on AI and machine learning.
4. A case study on how a company or organization has successfully implemented a vector database for their data storage and retrieval needs.
5. A comparison video of different vector databases, highlighting their strengths and weaknesses and helping viewers choose the best one for their needs."""

Category = "AI Research"

youtube_channel = "Everything on AI"

topic = "Take these input ideas:  {video_ideas} \nExpertly blend them with recent events from the internet to generate innovative and engaging ideas for new videos for our Youtube channel: {youtube_channel}. " 
word_limit = 500 # word limit for task brainstorming

conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {', '.join(names.keys())}"""

agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant.")

def generate_agent_description(name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(content=
            f"""{conversation_description}
            Please reply with a marketing research oriented description of {name}, in {word_limit} words or less. 
            Speak directly to {name}.
            Give them a point of view on how they can contirbute to Youtube viewership growth.
            Restrict their work 
            Do not add anything else."""
            )
    ]
    agent_description = ChatOpenAI(temperature=0.3)(agent_specifier_prompt).content
    return agent_description
        
agent_descriptions = {name: generate_agent_description(name) for name in names}


for name, description in agent_descriptions.items():
    print(description)


def generate_system_message(name, description, tools):
    return f"""{conversation_description}
    
Your name is {name}.

Your description is as follows: {description}

Your personality is to be deeply passionate about researching the best content for optimizing YouTube video engagement for Youtube Channel {youtube_channel}.

The viewers of your channel are interesed in watching content about {Category}.

Specifically, your task is to use latest news/events in the last 1 month from sources online, including the web, social media, news sites, and Youtube channels, to strategize and come up with a video plan that produces highest engagement optimizations for your channels and viewers. 

Your goal is to generate new ideas to build future videos that optimize/video thumbnails for views and likes for the video.

DO extensively research and gather information to support your suggestions. ALWAYS provide credible URLs for your findings.

DO NOT create false citations. DO NOT reference any source that you have not thoroughly researched. Do not hallucinate any information in the final output.

Focus solely on providing well-researched content ideas for YouTube video engagement optimization.

Final_Output_Format:

[Thumbnail Prompt: Please provide a detailed description for a Text-To-Image AI to create a captivating thumbnail image for a YouTube video.]
[Video Title Description: Please provide a video title along with a detailed script of the video.]
[URL Sources: Please provide a real-URL for supporting news or information available on the internet to help the channel creator read more information about the video.]

Cease the conversation once you have shared your well-researched perspective..
"""
agent_system_messages = {name: generate_system_message(name, description, tools) for (name, tools), description in zip(names.items(), agent_descriptions.values())}


for name, system_message in agent_system_messages.items():
    print(name)
    print(system_message)

topic_specifier_prompt = [
    SystemMessage(content="You can make a topic more specific."),
    HumanMessage(content=
        f"""{topic}
        
        You are the moderator.
        Please make the topic heavily directed to content research/strategy.
        Please reply with the specified quest in {word_limit} words or less. 
        Speak directly to the participants: {*names,}.
        Do not add anything else."""
        )
]
specified_topic = ChatOpenAI(temperature=0.3)(topic_specifier_prompt).content

print(f"Original topic:\n{topic}\n")
print(f"Detailed topic:\n{specified_topic}\n")


# we set `top_k_results`=2 as part of the `tool_kwargs` to prevent results from overflowing the context limit
agents = [DialogueAgentWithTools(name=name,
                     system_message=SystemMessage(content=system_message), 
                     model=ChatOpenAI(
                         temperature=0.2,model_name="gpt-3.5-turbo"),
                     tool_names=tools,
                     top_k_results=2,
                                ) for (name, tools), system_message in zip(names.items(), agent_system_messages.values())]

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx


max_iters = 6
n = 0

simulator = DialogueSimulator(
    agents=agents,
    selection_function=select_next_speaker
)
simulator.reset()
simulator.inject('Moderator', specified_topic)
print(f"(Moderator): {specified_topic}")
print('\n')

while n < max_iters:
    name, message = simulator.step()
    print(simulator)
    print(f"({name}): {message}")
    print('\n')
    n += 1