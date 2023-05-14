from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from flask import Flask, request, jsonify
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests




os.environ["SERPER_API_KEY"] = "d8c0597ad49019f1bbc962b915bddc32d96be5a8"
os.environ["GOOGLE_API_KEY"] = "UCg60QRUSvLZMF4zHv2ajBqA"
os.environ["GOOGLE_CSE_ID"] = "c691bdf65d73047ec"
app = Flask(__name__)
CORS(app)

@app.route('/agents', methods=['POST'])
def process_request():
    data = request.get_json()

    username = data.get('username')
    category = data.get('category')
    
    # Perform your internal operations here
    image_urls, title, description, script = trigger_function(username, category)

    print(image_urls, title, description, script)

    response = {
        'imageUrls': image_urls,
        'title': title,
        'description': description,
        'script': script
    }

    return jsonify(response)


def trigger_function(username, category):
    llm1 = OpenAI(temperature=0.3, openai_api_key = "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", model_name = "gpt-3.5-turbo")

    tools = load_tools(["google-serper"], llm=llm1)

    summaries = {'summary_1': 'Anthropic has launched a language model with a 100K context window, allowing for the analysis of entire books or long documents. The model can digest, summarize, and explain technical documents and perform complex tasks. The article discusses using the model to analyze podcasts and provide specific information based on questions asked. The accuracy of the model is attributed to the Assembly AI API. The paper introduces a new 100K context window model that can be used to avoid the need for custom Vector databases for smaller datasets. Links to the blog post and Assembly AI API are provided for further information.',
    'summary_2': 'Assembly AI has launched Lemur, a framework for transcribing speech using large language models. It can process up to 10 hours of audio content and includes intelligent segmentation, a fast Vector database, and reasoning steps. Lemur is accessible through a standard API and can generate task lists and answer questions about audio recordings. Early access is available by signing up to the waitlist.',
    'summary_3': 'Vector databases are becoming popular for storing unstructured data such as images, videos, and audio. They use algorithms to calculate numerical representations of the data, which are indexed and stored for fast retrieval and similarity search. They are useful for equipping language models with long-term memory, semantic search, similarity search for images, audio, or video data, and as a ranking and recommendation engine for online retailers. There are several vector databases available, including Pinecone, VV8, Chroma, Redis, CoolTrans, Milvus, and Vespa AI. The video provides an overview of vector databases and their applications.'}


    feedback = {'feedback_1': "Based on the comments, it seems like viewers are interested in the cost and capabilities of the Claude-v1-100k model. The YouTuber could create a follow-up video that dives deeper into the cost structure of using the model and explores its capabilities in more detail, including how it compares to other models like GPT4. Additionally, the YouTuber could address some of the technical issues mentioned in the comments, such as broken links and code that doesn't work.", 'feedback_2': 'Based on the comments, it seems like viewers are interested in learning more about the LeMUR API and how to use it. Therefore, the YouTuber could consider creating a tutorial video on how to access and utilize the LeMUR API in a project. This would provide value to viewers who are interested in machine learning and want to learn more about this specific API. Additionally, the YouTuber could consider showcasing some examples of projects that have successfully utilized the LeMUR API to inspire and motivate viewers to try it out themselves.', 'feedback_3': 'Based on the comments, it seems like viewers are interested in a more in-depth comparison of different vector databases, including their pros and cons, as well as their use cases and potential for generating vectors for multimodal data. Viewers also expressed interest in learning more about how to use vector databases for long-term memory in LLMs, as well as how indexes work and why vector databases are called databases. Therefore, a relevant and concise suggestion for the YouTuber for their next video would be to create a comprehensive guide to vector databases, covering topics such as their features, use cases, and limitations, as well as providing practical examples and tutorials on how to use them effectively. Additionally, the video could explore the latest trends and developments in vector databases, including new tools and techniques for generating and storing vectors, and provide insights into the future of this technology.'}

    video_ideas = f"""
    Summary of Last 3 Videos: {summaries['summary_1']} \n {summaries['summary_2']} \n {summaries['summary_3']}

    Feedback from Last 3 Videos: {feedback['feedback_1']} \n {feedback['feedback_1']} \n {feedback['feedback_3']}
    """

    Category = category

    topic = f"""Combine the input ideas: {video_ideas} with recent internet events to create innovative and engaging video ideas for a YouTube channel in the {Category} niche.""" 

    youtube_idea_researcher_prompt = f"""As a YouTube Idea Researcher passionately research the best content for optimizing engagement on the channel in the {Category} niche. Use the latest news/events from the past 2 months only, gathered from various online sources, to strategize a video plan for maximum engagement. Provie relevant sources for each idea that you consider """


    youtube_idea_researcher = initialize_agent(tools, llm=llm1, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    youtube_idea_researcher.agent.llm_chain.prompt.template = youtube_idea_researcher_prompt + youtube_idea_researcher.agent.llm_chain.prompt.template 

    internet_generated_ideas = youtube_idea_researcher.run(f"""Find the latest news events that occured in the last 2 months as a continuation from the topic:" + {topic} + ". The information should be original and not a copy of the topic but should resonate it well. Provide an output with an insightful summary of the key information online.""")

    print(internet_generated_ideas)


    llm2 = OpenAI(temperature=0.5, openai_api_key = "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", model_name = "gpt-4")

    tools = load_tools(["google-serper", "wikipedia", "llm-math"], llm=llm2)


    youtube_content_researcher_prompt = f"""As a YouTube Content Researcher AI agent, based on the following inputs: Ideas: {video_ideas} \n Online Research: {internet_generated_ideas} \nFor Youtube Channel in Niche: {Category} passionately brainstorm the best content for optimizing engagement on the channel in the {Category} niche. Use all inputs provided to you, to build a video on a single idea . Focus on well-researched content ideas for YouTube engagement optimization. Your final goal is to generate an output optimized for likes/views; with the video Thumbnail Prompt, Video Title, Video Description, Script of the Video"""

    youtube_content_researcher = initialize_agent(tools, llm=llm2, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    youtube_content_researcher.agent.llm_chain.prompt.template = youtube_content_researcher_prompt + youtube_content_researcher.agent.llm_chain.prompt.template 

    try:
        content_research_output =youtube_content_researcher.run(f"""Based on the following inputs: Ideas: {video_ideas} \n Online Research: {internet_generated_ideas} \nFor Youtube Channel in Niche: {Category}.\nGenerate a new original idea based on current events not present in the Ideas object but inspired by it and use it to create new video content which will make Youtube videos more engaging for viewers.
        The format of the Final answer should in the following:
        "
        [[Thumbnail Prompt:  A Detailed Prompt To Generate A Thumbnail Which Maximises For Views And Likes Of The Video],
        [Video Title: A Catchy and SEO optimised Youtube Video Title Which Maximises For Views And Likes Of The Video],
        [Video Description: A Detailed Video Description Which Maximises For Views And Likes Of The Video ],
        [Video_Script:A Comprehensive Scene-by-Scene Script Designed to Maximize Viewer Engagement and Watch Time]]
        """)

    except ValueError as e:
        content_research_output = str(e)
        content_research_output = extract_info(str(content_research_output))
        post_processing(content_research_output)

    content_research_output = extract_info(str(content_research_output))
    content_research_output = post_processing(content_research_output)
    return [content_research_output['Thumbnail'], content_research_output['Video Title'], content_research_output['Video Description'], content_research_output['Video Script']]

def extract_info(input_string):
    keys = ['Thumbnail Prompt', 'Video Title', 'Video Description', 'Video Script']
    info_dict = {}
        
    for i, key in enumerate(keys):
        start_index = input_string.find(key) + len(key) + 2
        end_index = input_string.find(keys[i + 1]) if i + 1 < len(keys) else len(input_string)
        info_dict[key] = input_string[start_index:end_index].strip()
        
    return info_dict

    


def get_lexica_data(query):
    url = "https://lexica.art/api/v1/search"
    params = {"q": query}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


def post_processing(final_dict):
    thumbnail_prompt = final_dict['Thumbnail Prompt']
    lexica_data = get_lexica_data(thumbnail_prompt)
    lex_data = []
    for i in range(min(4, len(lexica_data['images']))):  # Iterate through elements 0, 1, 2, and 3
        try:
            img_src = lexica_data['images'][i]['src']
            print(img_src)
            lex_data.append(img_src)  # Use append() method to add the image src to the list
        except:
            break
    final_dict['Thumbnail'] = lex_data
    return final_dict

if __name__ == '__main__':
    app.run(debug=True, port = 65501)

    
    
