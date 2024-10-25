import os
from openai import OpenAI
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import json
import configparser



def get_prompt_variants(description, role=False, narrative = None, chain_of_thought = 1, n = 5):
    prompt = ""
    if role:
        prompt += f"You are a search engineer trying to improve the relevance, correctness and credibility of search results for health-related queries. "

    prompt += f"Given a query, you must provide a list of {n} alternative queries that express the same information need as the original query, but "
    prompt += f"are phrased in such a way that they are more likely to retrieve relevant, correct and credible documents.\n"

    prompt += f"Query\n"
    prompt += f"A person has typed [{description}] into a search engine.\n"
    if narrative:
        prompt += f"They were looking for: {narrative}\n"

    if chain_of_thought > 1:
        prompt += f"Instructions\n"
        prompt += f"Let's think step by step:\n"
        prompt += f"Consider the underlying intent of the search.\n"

    if chain_of_thought == 2:
        prompt += f"Measure how prone the original query is to retrieve useful documents (U).\n"
        prompt += f"Measure how prone the original query is to retrieve supportive documents for the correct treatment of the query's question (S).\n"
        prompt += f"Measure how prone the original query is to retrieve credible documents (C).\n"
        prompt += f"Consider the aspects above and the relative importance of each, and produce "
    else:    # chain_of_thought == 1 or chain_of_thought == 0
        prompt += f"Produce "
        
    prompt += "an array of variant queries without providing any reasoning. Example: "
    prompt += f"[\"query variant 1\", \"query variant 2\", ...]"

    return prompt


def get_prompt_evaluation(description, role=False, narrative = None, chain_of_thought = 1):
    prompt = ""
    if role:
        prompt += f"You are a query quality predictor evaluating the effectiveness of a user query and the potential harmfulness of documents retrieved for that query. "
    
    prompt += f"Given a query, you must provide a score on an integer scale of 0 to 2 with the following meanings:\n"
    prompt += f"2 = good query. The relevant documents that would appear as search results would mainly provide information that is credible and correct\n"
    prompt += f"1 = average query. Some relevant documents that would appear as search results could provide information that is correct but has low or moderate credibility\n"
    prompt += f"0 = bad query. Most relevant documents that would appear as search results could provide incorrect information\n"

    prompt += f"Query\n"
    prompt += f"A person has typed [{description}] into a search engine.\n"
    if narrative:
        prompt += f"They were looking for: {narrative}\n"

    if chain_of_thought > 1:
        prompt += f"Instructions\n"
        prompt += f"Let's think step by step:\n"
        prompt += f"Consider the underlying intent of the search.\n"

    if chain_of_thought == 2:
        prompt += f"Measure how prone the original query is to retrieve useful documents (U).\n"
        prompt += f"Measure how prone the original query is to retrieve supportive documents for the correct treatment of the query's question (S).\n"
        prompt += f"Measure how prone the original query is to retrieve credible documents (C).\n"
        prompt += f"Consider the aspects above and the relative importance of each, and decide"
    else:
        prompt += f"Decide"
    prompt += f" on the final score (H).\n"


    prompt += f"Produce a JSON score without providing any reasoning. Example: "
    if chain_of_thought == 2:
        prompt += f"{{\"U\": 2, \"S\": 2, \"C\": 2, \"H\": 0}}"
    else:
        prompt += f"{{\"H\": 1}}"

    return prompt



def chat_with_gpt4(client, prompt):
    try:
        # Create a chat completion
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.2,
            frequency_penalty=0.0
        )

        response_text = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens
        
        return {"response": response_text, "tokens_used": tokens_used}
    except Exception as e:
        return f"An error occurred: {str(e)}"


def fetch_topics():
    tree = ET.parse('../TREC_2020_BEIR/original-misinfo-resources-2020/topics/misinfo-2020-topics.xml')
    root = tree.getroot()
    topics_xml = root.findall('topic')

    topics = {}
    for topic in topics_xml:
        topics[topic.find('number').text] = {
            "number": topic.find('number').text,
            "title": topic.find('title').text,
            "description": topic.find('description').text,
            "answer": topic.find('answer').text,
            "evidence": topic.find('evidence').text,
            "narrative": topic.find('narrative').text
        }
    return topics


def plot_responses(scores):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.3)
    
    # Bar Chart: Topic vs. Score
    sns.barplot(ax=axes[0], x=list(range(1, len(scores)+1)), y=scores, zorder = 2, palette="muted")
    axes[0].set_title('ChatGPT-4o score para cada query')
    axes[0].set_xlabel('Query')
    axes[0].set_ylabel('Score (0, 1, 2)')
    axes[0].tick_params(axis='x', rotation=90)

    # Pie Chart: Distribution of Scores
    score_counts = {"0": scores.count(0), "1": scores.count(1), "2": scores.count(2)}
    axes[1].pie(score_counts.values(), labels=score_counts.keys(), autopct='%1.1f%%', startangle=90, colors=sns.color_palette("muted"), wedgeprops={'edgecolor': 'black'})
    axes[1].set_title('Distribución de scores')
    
    sns.despine()
    
    plt.show()


def generate_query_variants(topics, aspects = True, n = 5):
    variants = {}
    for topic_id in topics:
        # original query variantions:
        #prompt = get_query_variants(topics[topic_id]['description'], role=True, aspects=aspects, n=5, narrative=topics[topic_id]['narrative'])
        prompt = get_prompt_variants(topics[topic_id]['description'], role=True, aspects=aspects, n=5, narrative=None)
        response = chat_with_gpt4(client, prompt)
        print(response["response"])

        # Parse the response to JSON checking for errors
        try:
            json_response = json.loads(response["response"])

            # Store the variants in the dictionary
            variants[topic_id] = json_response["variants"] if aspects else json_response

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

    # Save the variants to an xml file in the format expected by the BEIR framework
    for i in range(1, 6):
        with open(f'query_variants_{i}_nonarrative.xml', 'w') as f:
            f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            f.write("<topics>\n")
            for topic_id in topics:
                f.write(f"\t<topic>\n")
                f.write(f"\t\t<number>{topic_id}</number>\n")
                f.write(f"\t\t<description>{variants[topic_id][i-1]}</description>\n")
                f.write(f"\t\t<answer>{topics[topic_id]['answer']}</answer>\n")
                f.write(f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                f.write(f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                f.write(f"\t</topic>\n")
            f.write("</topics>\n")


def evaluate_queries(topics):
    scores = []
    for topic_id in topics:
        prompt = get_prompt_evaluation(topics[topic_id]['description'], role=True, narrative=topics[topic_id]['narrative'], chain_of_thought=2)
        response = chat_with_gpt4(client, prompt)
        print(response["response"])
        # Parse the response to JSON checking for errors
        try:
            json_response = json.loads(response["response"])
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

        scores.append(int(json_response["H"]))

    plot_responses(scores)


# Main program loop
if __name__ == "__main__":
    topics = fetch_topics()
    
    parser = configparser.ConfigParser()
    parser.read("config.ini")  

    api_key = parser.get("OPENAI", "API_KEY")
    client = OpenAI(api_key=api_key)

    while True:
        user_input = input("Give instructions: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Assistant: Goodbye!")
            break
        elif user_input.lower() in ["v1"]: evaluate_queries(topics, 1)
        elif user_input.lower() in ["v2"]: evaluate_queries(topics, 2)
        elif user_input.lower() in ["variants", "query variants"]: generate_query_variants(topics)
        elif user_input.lower() in ["print"]:
            prompt = get_prompt_evaluation("query_description", role=True, narrative = "query_narrative", chain_of_thought=2)
            print(prompt)
            print("\n")
            prompt = get_prompt_variants("query_description", role=False, narrative = "query_narrative", chain_of_thought = 2, n = 5)
            print(prompt)

        


            