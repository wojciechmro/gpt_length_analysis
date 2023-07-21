# LIBRARIES
import openai  # wrapper for OpenAI API
import config.key as key  # key.py file with API key
import tiktoken  # tokenizer for OpenAI API
import pandas as pd  # data manipulation
import time  # retrying API calls

# CONSTANTS
openai.api_key = key.openai_api_key  # put your API key here
ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")  # choose model tokenizer
RETRY_DELAY_SECONDS = 10  # seconds to wait before retrying API call

# COLORS FOR PRINTING
BLUE = "\033[94m"
RESET = "\033[0m"


def generate_responses(
    n: int,
    prompt: str,
    filename_prefix: str,
    filename_suffix: str,
    model="gpt-3.5-turbo",
    folder_type="experimental",
) -> None:
    """
    Generates n responses from OpenAI. Saves to CSV file.

    Parameters
    ----------
    n : int
        Number of responses to generate.

    prompt : str
        Prompt sent to OpenAI.

    filename_prefix : str
        Used as first element of filename. e.g. "email_*.csv".

    filename_suffix : str
        Used as second element of filename. e.g. "*_short.csv".

    model : str, optional
        Model to use for generation. Defaults to "gpt-3.5-turbo".

    folder_type : str, optional
        Used as element of folder name. e.g. "raw_*_group". Defaults to "experimental". Other option is "control".
    """

    text, chars, words, tokens = [], [], [], []
    success_counter, error_counter = 0, 0

    while success_counter < n:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )["choices"][0]["message"]["content"]

            text.append(response)
            chars.append(len(response))
            words.append(len(response.split()))
            tokens.append(len(ENC.encode(response)))

            success_counter += 1

        except Exception as e:
            error_counter += 1

            print(f"Error generating response {success_counter}: {e}")
            time.sleep(RETRY_DELAY_SECONDS)

        print(
            f"{BLUE}Responses gathered: {str(success_counter)}  |  Working on: {filename_prefix}_{filename_suffix}.csv  |  Errors occured: {str(error_counter)}{RESET}"
        )

    # DataFrame from gathered lists
    data = {"text": text, "chars": chars, "words": words, "tokens": tokens}
    df = pd.DataFrame(data)

    # to CSV
    filename = f"{filename_prefix}_{filename_suffix}.csv"
    df.to_csv(f"data/raw_{folder_type}_group/{filename}", index=False)

    # log
    print(f"Filename {filename} was created!\n")
