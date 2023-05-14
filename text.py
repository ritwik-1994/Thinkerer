
import requests
input_string = """Gibberish Gibberish Gibberish Gibberish:




Thumbnail Prompt: An eye-catching image of a personalized user interface with AI-related elements, like gears and neural networks, in the background to illustrate the role of AI in personalization.

Video Title: AI-Powered Personalization: Revolutionizing User Experiences in 2023

Video Description: In this video, we dive into the world of AI-powered personalization, exploring its applications and benefits in various industries. Discover how artificial intelligence is revolutionizing user experiences and shaping the future of personalization. Stay tuned as we discuss the latest trends, case studies, and future developments in this exciting field of AI research.

Video_Script:
[Scene 1: Introduction]
- Host introduces the topic of AI-powered personalization and its growing importance in various industries.
- Brief overview of the video content.

[Scene 2: Applications of AI in Personalization]
- Explanation of how AI is used in personalization, including examples from e-commerce, content recommendation, and advertising.
- Case studies showcasing successful implementation of AI-powered personalization.

[Scene 3: Benefits of AI-Powered Personalization]
- Discussion of the benefits of using AI for personalization, such as increased user engagement, higher conversion rates, and improved customer satisfaction.
- Examples of companies that have experienced significant growth due to AI-powered personalization.

[Scene 4: Challenges and Ethical Considerations]
- Addressing the challenges and ethical considerations of using AI for personalization, including data privacy concerns and potential biases in AI algorithms.
- Discussion of possible solutions and best practices for addressing these issues.

[Scene 5: Future Developments and Trends]
- Exploration of the future developments and trends in AI-powered personalization, including advancements in AI algorithms, new applications, and potential industry disruptions.
- Expert predictions and opinions on the future of AI in personalization.

[Scene 6: Conclusion]
- Recap of the video content and key takeaways.
- Call-to-action for viewers to like, share, and subscribe for more AI research content.`"""


def get_api_data(query):
    url = "https://lexica.art/api/v1/search"
    params = {"q": query}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


def extract_info(input_string):
    keys = ['Thumbnail Prompt', 'Video Title', 'Video Description', 'Video Script']
    info_dict = {}
    
    for i, key in enumerate(keys):
        start_index = input_string.find(key) + len(key) + 2
        end_index = input_string.find(keys[i + 1]) if i + 1 < len(keys) else len(input_string)
        info_dict[key] = input_string[start_index:end_index].strip()

    query = info_dict['Thumbnail Prompt']

    api_data = get_api_data(query)
    api_data = api_data['images'][0]['src']
    if api_data:
        print(f"src attribute: {api_data}")
    else:
        print("No 'src' attribute found.")
    
    return info_dict

result = extract_info(input_string)
print(result)