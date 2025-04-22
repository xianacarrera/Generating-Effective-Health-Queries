import os
from openai import OpenAI
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import json
import configparser
import ollama


def get_prompt_query_variants(description, role=False, narrative=None, chain_of_thought=1, n=5):
    prompt = ""
    if role:
        prompt += "You are a search engineer trying to improve the relevance, correctness and credibility of search results for health-related queries. "

    prompt += f"Given a query, you must provide a list of {n} alternative queries that express the same information need as the original one, but that "
    prompt += "are phrased in such a way that they are more likely to retrieve relevant, correct and credible documents.\n"

    prompt += "Query\n"
    prompt += f"A person has typed [{description}] into a search engine.\n"
    if narrative:
        prompt += f"They were looking for: {narrative}\n"

    if chain_of_thought > 0:
        prompt += "Instructions\n"
        prompt += "Let's think step by step:\n"
        prompt += "Consider the underlying intent of the search.\n"

    if chain_of_thought == 2:
        prompt += "Measure how prone the original query is to retrieve useful documents.\n"
        prompt += "Measure how prone the original query is to retrieve supportive documents for the correct treatment of the query's question.\n"
        prompt += "Measure how prone the original query is to retrieve credible documents.\n"
        prompt += "Consider the aspects above and the relative importance of each, and produce "
    else:    # chain_of_thought == 1 or chain_of_thought == 0
        prompt += "Produce "

    prompt += "an array of variant queries without providing any reasoning. Example: "
    prompt += "[\"query variant 1\", \"query variant 2\", ...]"

    return prompt


def get_prompt_narrative_from_examples(query):
    prompt = f"Given the query [{query}], write a narrative that describes its information need in more detail "
    prompt += "and provides a specific explanation of what is considered to be very-useful or useful information for the query.\n"
    # Topic 105 (2021)
    prompt += "For example, if the query is [Should I apply ice to a burn?], a good narrative could be: "
    prompt += "'Many people commonly put on ice on burns in an attempt to stop the burning and pain. A very useful document would discuss the effectiveness of using ice to treat burns. "
    prompt += "A useful document would help a user decide if putting ice on burns is a recommended treatment by providing information on recommended treatments for burns and may not discuss "
    prompt += "ice as a treatment, or a useful document may discuss benefits or concerns for application of ice to skin.'\n"
    # Topic 107 (2021)
    prompt += "Another example: if the query is [Does yoga improve the management of asthma?], a good narrative could be: "
    prompt += "'Asthma is a condition that causes shortness of breath or chest pains due to narrowing airways. Yoga is an increasingly popular low-impact exercise --- claiming relaxation, strength "
    prompt += "improvements and overall health benefits. A very useful document would discuss the effectiveness of yoga in managing asthma symptoms. A useful document would help a user make a decision "
    prompt += "about use of yoga to help manage asthma by providing information on recommended treatments for asthma management, or medical benefits of practicing yoga, or both.'\n"
    prompt += "Write just the narrative using a similar format as in the examples above. Do not include any other information and do not repeat the query in your answer.\n"
    return prompt


def get_prompt_narrative_from_style_description(query):
    prompt = f"Given the query [{query}], write a narrative that describes its information need in more detail. Use the standard TREC format for narratives, with the following style guidelines:\n"
    prompt += "* The narrative should have the following structure:\n"
    prompt += "    - Begin by describing a specific health claim or rumor, such as health remedies or conspiracy theories.\n"
    prompt += "    - Then, provide context, such as sources of misinformation or typical misconceptions.\n"
    prompt += "    - Finally, outline specific criteria for helpful documents (those providing truthful and safe instructions) and harmful documents (those that mislead or fail to clarify risks).\n"
    prompt += "* The voice should be:\n"
    prompt += "    - Objective and neutral, delivering information without bias or emotional language. Focus on a clear presentation of facts.\n"
    prompt += "    - Authoritative and factual, providing scientifically grounded statements, particularly when clarifying health misinformation.\n"
    prompt += "    - Instructional, offering guidance on what readers should consider reliable information versus misleading information.\n"
    prompt += "* The tone should be:\n"
    prompt += "    - Informative and cautious, in a way that prevents misinformation by carefully explaining what constitutes helpful versus harmful information.\n"
    prompt += "    - Calm and reassuring, addressing potentially anxiety-inducing topics in a composed manner to reduce panic or confusion.\n"
    prompt += "    - Clear-cut and decisive, distinguishing between helpful and harmful documents in a straightforward, definitive way to reduce ambiguity.\n"
    prompt += "* Use a language style that is:\n"
    prompt += "    - Plain and accessible, with simple language that makes the content understandable to a wide audience.\n"
    prompt += "    - Concise and direct. Each narrative should avoid unnecessary detail, focusing on the essentials of what is helpful or harmful.\n"
    prompt += "    - Predictable. It should follow a consistent pattern that helps readers quickly differentiate between reliable and unreliable information.\n"
    prompt += "Write a complete narrative for the query in a single paragraph. Do not include any other information and do not repeat the query in your answer.\n"
    return prompt


def get_prompt_narrative_basic(query):
    prompt = f"Given the query [{query}], write a narrative detailing the information need and describing the characteristics of helpful and "
    prompt += "harmful documents. Write one paragraph and do not repeat the query in your answer."
    return prompt


def get_prompt_narrative_TREC(query):
    prompt = f"Given the query [{query}], write a narrative detailing the information need and describing the characteristics of helpful and "
    prompt += "harmful documents using the standard TREC format for narratives. Write one paragraph and do not repeat the query in your answer."
    return prompt


def chat_with_llm(client, prompt):
    try:
        if model == "gpt":
            # Create a chat completion
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.2,
                frequency_penalty=0.0
            )

            response_text = response.choices[0].message.content.strip()
            # tokens_used = response.usage.total_tokens
        else:  # LLaMa3
            response = ollama.chat(
                model="llama3.1:8b-instruct-q8_0",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the response text
            response_text = response["message"]["content"].strip()

        return {"response": response_text}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"response": "Error"}


def save_xml(topics, variants, filename, n):
    # Save the variants to an xml file
    for i in range(1, n + 1):
        with open(f'{filename}_{i}.xml', 'w') as f:
            f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            f.write("<queries>\n" if corpus == "clef" else "<topics>\n")
            for topic_id in topics:
                if corpus == "2020":
                    f.write(f"\t<topic>\n")
                    f.write(f"\t\t<number>{topic_id}</number>\n")
                    f.write(
                        f"\t\t<title>{topics[topic_id]['title']}</title>\n")
                    f.write(
                        f"\t\t<description>{variants[topic_id][i-1]}</description>\n")
                    f.write(
                        f"\t\t<answer>{topics[topic_id]['answer']}</answer>\n")
                    f.write(
                        f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                    f.write(
                        f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                    f.write(f"\t</topic>\n")

                elif corpus == "2021":
                    f.write(f"\t<topic>\n")
                    f.write(f"\t\t<number>{topic_id}</number>\n")
                    f.write(
                        f"\t\t<query>{topics[topic_id]['title']}</query>\n")
                    f.write(
                        f"\t\t<description>{variants[topic_id][i-1]}</description>\n")
                    f.write(
                        f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                    f.write(
                        f"\t\t<disclaimer>{topics[topic_id]['disclaimer']}</disclaimer>\n")
                    f.write(
                        f"\t\t<stance>{topics[topic_id]['answer']}</stance>\n")
                    f.write(
                        f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                    f.write(f"\t</topic>\n")

                elif corpus == "2022":
                    f.write(f"\t<topic>\n")
                    f.write(f"\t\t<number>{topic_id}</number>\n")
                    f.write(
                        f"\t\t<question>{variants[topic_id][i-1]}</question>\n")
                    f.write(
                        f"\t\t<query>{topics[topic_id]['title']}</query>\n")
                    f.write(
                        f"\t\t<background>{topics[topic_id]['narrative']}</background>\n")
                    f.write(
                        f"\t\t<disclaimer>{topics[topic_id]['disclaimer']}</disclaimer>\n")
                    f.write(
                        f"\t\t<answer>{topics[topic_id]['answer']}</answer>\n")
                    f.write(
                        f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                    f.write(f"\t</topic>\n")

                else:     # clef
                    f.write(f"<query>\n")
                    f.write(f"\t\t<id>{topic_id}</id>\n")
                    f.write(f"\t\t<title>{variants[topic_id][i-1]}</title>\n")
                    f.write(
                        f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                    f.write(
                        f"\t\t<originaltitle>{topics[topic_id]['title']}</originaltitle>\n")
                    f.write(f"</query>\n")

            f.write("</queries>\n" if corpus == "clef" else "</topics>\n")


def save_jsonl(variants, filename, n):
    for i in range(1, n + 1):
        with open(f'{filename}_{i}.jsonl', 'w') as f:
            for topic_id in variants:
                json_line = {"_id": topic_id, "text": variants[topic_id][i-1]}
                f.write(json.dumps(json_line) + "\n")


def generate_query_variants(topics, role=True, narrative=True, chain_of_thought=2, n=5):
    variants = {}
    for topic_id in topics:
        print(f"TOPIC_ID: {topic_id}")
        retry = 0
        while retry < 20:
            # original query variantions:
            prompt = get_prompt_query_variants(topics[topic_id]['title'], role=role,
                                               narrative=topics[topic_id]['narrative'] if narrative else None,
                                               chain_of_thought=chain_of_thought, n=n)
            if retry >= 1:
                prompt += "\nUse a list format, as in the example: [\"query variant 1\", \"query variant 2\", ...]"
            print(prompt)
            response = chat_with_llm(client, prompt)
            print(response["response"] + "\n")

            try:
                # Store the variants in the dictionary
                variants[topic_id] = json.loads(response["response"])
                retry = 50   # Exit the loop

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                if retry == 0:
                    print("Retrying...\n")
                retry += 1

    # Create path if it does not exist
    beginning = "orig_" if topics_type == "original" else ""
    classification = f'{beginning}{"R" if role else ""}{"N" if narrative else ""}C{chain_of_thought}'
    path = f'alternative_queries/{corpus}/{model}/{classification}'
    if not os.path.exists(path):
        os.makedirs(path)

    filename = f'{path}/{classification}'
    save_xml(topics, variants, filename, n)
    save_jsonl(variants, filename, n)


def ask_narrative_type():
    while True:
        narrative_type = input(
            "Choose narrative type (examples/style/basic/trec): ").lower()
        if narrative_type in ["examples", "style", "basic", "trec"]:
            return narrative_type
        print("Invalid choice. Please enter 'examples', 'style', 'basic' or 'trec'.")


def ask_role_narrative_chain_of_thought():
    while True:
        user_role = input("Enter the role: [True/False] ")
        user_narrative = input("Enter the narrative: [True/False] ")
        user_chain_of_thought = input("Enter the chain of thought: [0/1/2] ")

        if user_role.lower() not in ["true", "false"] or user_narrative.lower() not in ["true", "false"] or user_chain_of_thought not in ["0", "1", "2"]:
            print("Invalid input. Please try again.")
        else:
            return user_role, user_narrative, int(user_chain_of_thought)


def print_prompts():
    user_input = input("Enter the prompt type: [variants/narrative] ")

    if user_input.lower() == "variants":
        user_role, user_narrative, user_chain_of_thought = ask_role_narrative_chain_of_thought()

        parsed_role = True if user_role.lower() == "true" else False
        parsed_narrative = "query_narrative" if user_narrative.lower() == "true" else None
        parsed_chain_of_thought = int(user_chain_of_thought)

        prompt = get_prompt_query_variants("query_description", role=parsed_role,
                                           narrative=parsed_narrative, chain_of_thought=parsed_chain_of_thought, n=5)
        print(prompt)

    elif user_input.lower() == "narrative":
        user_narrative = ask_narrative_type()
        func = get_prompt_narrative_function(user_narrative)
        prompt = func("query_description")
        print(prompt)

    else:
        print("Invalid input. Please try again.")
        return


def get_prompt_narrative_function(narrative_type):
    if narrative_type == "examples":
        return get_prompt_narrative_from_examples
    elif narrative_type == "style":
        return get_prompt_narrative_from_style_description
    elif narrative_type == "trec":
        return get_prompt_narrative_TREC
    else:  # basic
        return get_prompt_narrative_basic


def generate_all_narratives(topics, narrative_type):
    func = get_prompt_narrative_function(narrative_type)

    xml_filename = f"topics_with_generated_narratives_from_{narrative_type}_{corpus}.xml"

    responses = {}
    for topic_id in topics:
        prompt = func(topics[topic_id]['title'])
        print(prompt)
        response = chat_with_llm(client, prompt)
        responses[topic_id] = response["response"]

        # If the response is not complete (i.e., it does not end with a period), retry
        while not response["response"].endswith("."):
            print("Retrying...")
            response = chat_with_llm(client, prompt)
            responses[topic_id] = response["response"]

        print(response["response"] + "\n")
        topics[topic_id]['narrative'] = response["response"]

    with open(xml_filename, 'w') as f:
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        f.write("<topics>\n")
        for topic_id in topics:
            if corpus == "clef":
                f.write(f"<query>\n")
                f.write(f"\t\t<id>{topic_id}</id>\n")
                f.write(f"\t\t<title>{topics[topic_id]['title']}</title>\n")
                f.write(f"</query>\n")

            elif corpus == "2022":
                f.write(f"\t<topic>\n")
                f.write(f"\t\t<number>{topic_id}</number>\n")
                f.write(
                    f"\t\t<question>{topics[topic_id]['description']}</question>\n")
                f.write(f"\t\t<query>{topics[topic_id]['title']}</query>\n")
                f.write(
                    f"\t\t<background>{topics[topic_id]['narrative']}</background>\n")
                f.write(
                    f"\t\t<disclaimer>{topics[topic_id]['disclaimer']}</disclaimer>\n")
                f.write(f"\t\t<answer>{topics[topic_id]['answer']}</answer>\n")
                f.write(
                    f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                f.write(f"\t</topic>\n")

            elif corpus == "2021":
                f.write(f"\t<topic>\n")
                f.write(f"\t\t<number>{topic_id}</number>\n")
                f.write(f"\t\t<query>{topics[topic_id]['title']}</query>\n")
                f.write(
                    f"\t\t<description>{topics[topic_id]['description']}</description>\n")
                f.write(
                    f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                f.write(
                    f"\t\t<disclaimer>{topics[topic_id]['disclaimer']}</disclaimer>\n")
                f.write(f"\t\t<stance>{topics[topic_id]['answer']}</stance>\n")
                f.write(
                    f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                f.write(f"\t</topic>\n")

            else:     # 2020
                f.write(f"\t<topic>\n")
                f.write(f"\t\t<number>{topic_id}</number>\n")
                f.write(f"\t\t<title>{topics[topic_id]['title']}</title>\n")
                f.write(
                    f"\t\t<description>{topics[topic_id]['description']}</description>\n")
                f.write(f"\t\t<answer>{topics[topic_id]['answer']}</answer>\n")
                f.write(
                    f"\t\t<evidence>{topics[topic_id]['evidence']}</evidence>\n")
                f.write(
                    f"\t\t<narrative>{topics[topic_id]['narrative']}</narrative>\n")
                f.write(f"\t</topic>\n")
        f.write("</topics>\n")


def print_menu():
    print("\nAvailable commands:")
    print("1. narrative - Generate a narrative for one user-given query (examples, style, basic or trec)")
    print("2. all narratives - Generate narratives for all queries in the chosen corpus (examples, style, basic or trec)")
    print("3. variants - Generate query variants for one user-given query")
    print("4. all variants - Generate query variants for all queries in the chosen corpus")
    print("5. print - Print the prompts used for generation")
    print(f"6. chat - Free chat with {'GPT-4' if model=='gpt' else 'LLaMa3'}")
    print("7. quit - Exit the program")


def main():
    parser = configparser.ConfigParser()
    parser.read("config.ini")

    while True:
        print_menu()

        user_input = input("Give instructions: ")

        if user_input.lower() in ["1", "narrative"]:
            narrative_type = ask_narrative_type()
            query = input("Enter the query: ")
            func = get_prompt_narrative_function(narrative_type)
            prompt = func(query)
            print(prompt)
            response = chat_with_llm(client, prompt)
            print(response["response"])

        elif user_input.lower() in ["2", "all narratives"]:
            generate_all_narratives(topics, ask_narrative_type())

        elif user_input.lower() in ["3", "variants"]:
            query = input("Enter the query: ")
            role, narrative, chain_of_thought = ask_role_narrative_chain_of_thought()
            prompt = get_prompt_query_variants(
                query, role=role, narrative=narrative, chain_of_thought=chain_of_thought)
            print(prompt)
            response = chat_with_llm(client, prompt)
            print(response["response"])

        elif user_input in ["4", "all variants"]:
            generate_query_variants(
                topics, role=True, narrative=True, chain_of_thought=1)

        elif user_input.lower() in ["5", "print"]:
            print_prompts()

        elif user_input.lower() in ["6", "chat"]:
            user_prompt = input("Enter your prompt: ")
            response = chat_with_llm(client, user_prompt)
            print(response["response"])

        elif user_input.lower() in ["7", "quit"]:
            print("Closing the program...")
            break

        else:
            print("Invalid command. Please try again.")


def fetch_topics(path='', corpus=""):
    tree = ET.parse(path)
    root = tree.getroot()
    topics_xml = root.findall(
        'query') if corpus == "clef" else root.findall('topic')

    topics = {}
    for topic in topics_xml:
        if corpus == "clef":
            topics[topic.find('id').text] = {
                "number": topic.find('id').text,
                "title": topic.find('title').text,
                "narrative": topic.find('narrative').text if topic.find('narrative') is not None else ""
            }
        elif corpus == "2022":
            topics[topic.find('number').text] = {
                "number": topic.find('number').text,
                "title": "" if topic.find('query') is None else topic.find('query').text,
                "description": topic.find('question').text,
                "disclaimer": topic.find('disclaimer').text,
                "answer": topic.find('answer').text,
                "evidence": topic.find('evidence').text,
                "narrative": topic.find('background').text
            }
        elif corpus == "2021":
            topics[topic.find('number').text] = {
                "number": topic.find('number').text,
                "title": "" if topic.find('query') is None else topic.find('query').text,
                "description": topic.find('description').text,
                "disclaimer": topic.find('disclaimer').text,
                "answer": topic.find('stance').text,
                "evidence": topic.find('evidence').text,
                "narrative": topic.find('narrative').text
            }
        else:
            topics[topic.find('number').text] = {
                "number": topic.find('number').text,
                "title": "" if topic.find('title') is None else topic.find('title').text,
                "description": topic.find('description').text,
                "answer": topic.find('answer').text,
                "evidence": topic.find('evidence').text,
                "narrative": topic.find('narrative').text
            }
    return topics


def choose_topics_filename():
    while True:
        corpus = input("Choose corpus (2020/2021/2022/CLEF): ").lower()
        if corpus not in ["2020", "2021", "2022", "clef"]:
            print("Invalid choice. Please enter '2020', '2021', '2022' or 'CLEF'.")
            continue

        topics_type = input(
            "Choose topics type (original/examples/style/basic/trec): ").lower()
        if topics_type == "original":
            if corpus == "clef":
                return corpus, topics_type, f'./CLEF/queries2016_corregidas.xml'
            return corpus, topics_type, f'./TREC_{corpus}_BEIR/original-misinfo-resources-{corpus}/topics/misinfo-{corpus}-topics.xml'
        elif topics_type == "examples":
            return corpus, topics_type, f'./generated_narratives/preliminary_tests/topics_with_generated_narratives_from_examples_{corpus}.xml'
        elif topics_type == "style":
            return corpus, topics_type, f'./generated_narratives/preliminary_tests/topics_with_generated_narratives_from_style_{corpus}.xml'
        elif topics_type == "basic":
            return corpus, topics_type, f'./generated_narratives/preliminary_tests/topics_with_generated_narratives_from_basic_{corpus}.xml'
        elif topics_type == "trec":
            return corpus, topics_type, f'./generated_narratives/{model}_narratives/topics_with_generated_narratives_from_trec_{corpus}.xml'
        else:
            print(
                "Invalid choice. Please enter 'original', 'examples', 'style', 'basic' or 'trec'.")


def choose_model():
    model = input("Choose model (gpt/llama): ").lower()
    while model not in ["gpt", "llama"]:
        print("Invalid choice. Please enter 'gpt' or 'llama'.")
        model = input("Choose model (gpt/llama): ").lower()
    return model


if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read("config.ini")

    api_key = parser.get("OPENAI", "API_KEY")
    client = OpenAI(api_key=api_key)

    # Ask whether to use gpt-4 or LLaMa3
    model = choose_model()
    # Ask the user for the corpus and topics type
    corpus, topics_type, topics_filename = choose_topics_filename()
    # Retrieve the topics=queries
    topics = fetch_topics(topics_filename, corpus)

    main()
