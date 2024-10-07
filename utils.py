import pandas as pd
from dotenv import load_dotenv


# Function to load and validate configuration settings
def load_config():
    # Load environment variables from a .env file
    load_dotenv()


def remove_nul_chars_from_string(s):
    """Remove NUL characters from a single string."""
    return s.replace('\x00', '')


def remove_nul_chars_from_run_data(run_data):
    """Iterate over all fields of RunData to remove NUL characters."""
    for run in run_data:
        run.reference_question = remove_nul_chars_from_string(run.reference_question)
        run.reference_answer = remove_nul_chars_from_string(run.reference_answer)
        run.llm_answer = remove_nul_chars_from_string(run.llm_answer)
        run.llm_context = [remove_nul_chars_from_string(context) for context in run.llm_context]


def make_get_llama_response(query_engine):
    def get_llama_response(prompt):
        # print(prompt)
        response = query_engine.query(prompt)
        context = []
        for x in response.source_nodes:
            # Initialize context string with the text of the node
            node_context = x.text
            # Check if 'window' metadata exists and append it to the context
            if 'window' in x.metadata:
                node_context += "\n\nWindow Context:\n" + x.metadata['window']
            context.append(node_context)
        return {
            "llm_answer": response.response,
            "llm_context_list": context
        }
    return get_llama_response


def run_experiment(experiment_name, query_engine, scorer, benchmark,
                   validate_api, project_key, upload_results=True, runs=5):
    # List to store results dictionaries
    results_list = []

    for i in range(runs):
        get_llama_response_func = make_get_llama_response(query_engine)
        run = scorer.score(benchmark,
                           get_llama_response_func,
                           callback_parallelism=1,
                           scoring_parallelism=1)
        print(f"{experiment_name} Run {i+1} Overall Scores:", run.overall_scores)
        remove_nul_chars_from_run_data(run.run_data)

        # Add this run's results to the list
        results_list.append({'Run': i+1, 'Experiment': experiment_name, 'OverallScores': run.overall_scores})

        if upload_results:
          project_key="b9df0eaf-5f47-4639-b3db-25487214e8ab"
          validate_api.upload_run(project_key, run=run, run_metadata={"approach": experiment_name, "run_number": i+1})
        else:
          print(f"Skipping upload for {experiment_name} Run {i+1}.")

    # Create a DataFrame from the list of results dictionaries
    results_df = pd.DataFrame(results_list)

    # Return the DataFrame containing all the results
    return results_df
