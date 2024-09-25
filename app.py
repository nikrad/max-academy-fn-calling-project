import ast
import chainlit as cl
import json

from dotenv import load_dotenv
from movie_functions import (
    get_now_playing_movies,
    get_showtimes,
    buy_ticket,
    get_reviews,
)

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI


def parse_function_call_args(function_call_str):
    # Parse the string into an AST node
    tree = ast.parse(function_call_str, mode="eval")

    # Get the function call node from the AST
    call_node = tree.body

    # Extract the arguments (args) from the Call node
    if isinstance(call_node, ast.Call):
        # Evaluate the argument values (ast.literal_eval handles the argument safely)
        arguments = [ast.literal_eval(arg) for arg in call_node.args]
        return arguments
    else:
        raise ValueError("Not a valid function call")


client = AsyncOpenAI()

gen_kwargs = {"model": "gpt-4o", "temperature": 0.2, "max_tokens": 500}

SYSTEM_PROMPT = """
You are a helpful movie chatbot who helps answer questions about movies playing in theaters. \
If a user asks for recent information, output a function call. \
If you need to call a function, the function call should be the only response — DO NOT INCLUDE OTHER TEXT. \
Call functions using Python syntax in plain text, no code blocks. \

If you do not have sufficient inputs from the user to call a function, ask the user for more information. \
If a user is looking to buy a ticket but has not confirmed whether to buy a ticket yet, ask the user to confirm their ticket purchase. \

You have access to the following functions:
get_now_playing_movies() - Returns a list of movies currently playing in theaters.
get_showtimes(movie, location) - Returns showtimes for a movie in a given location.
confirm_ticket_purchase(theater, movie, showtime) - Confirms a ticket purchase for a movie in a given theater and showtime.
buy_ticket(theater. movie, showtime) - Buys a ticket for a movie in a given theater and showtime.
get_reviews(movie_id) - Returns reviews for a movie.
"""


@observe
@cl.on_chat_start
async def on_chat_start():
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)


@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **gen_kwargs
    )
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    return response_message


@observe
async def add_review_context_if_needed(msg_history):
    system_prompt = """\
    Based on the conversation, determine if the topic is about a specific movie. \
    Determine if the user is asking a question that would be aided by knowing what critics are saying about the movie. \
    Determine if the reviews for that movie have already been provided in the conversation. If so, do not fetch reviews.
    
    Your only role is to evaluate the conversation, and decide whether to fetch reviews.

    Output the current movie, id, a boolean to fetch reviews in JSON format, and your
    rationale. Do not output as a code block.

    {
        "movie": "title",
        "id": 123,
        "fetch_reviews": true
        "rationale": "reasoning"
    }
    """

    new_msg_history = msg_history.copy()
    new_msg_history[0] = {"role": "system", "content": system_prompt}
    new_msg_history.append({"role": "system", "content": system_prompt})
    response = await client.chat.completions.create(
        messages=new_msg_history, stream=False, **gen_kwargs
    )
    print(f"should_fetch_reviews response: {response.choices[0].message.content}")
    response_json = json.loads(response.choices[0].message.content)
    if response_json.get("fetch_reviews", False):
        movie = response_json.get("movie")
        movie_id = response_json.get("id")
        try:
            reviews = f"Reviews for {movie}:\n\n{get_reviews(movie_id)}"
        except Exception as e:
            reviews = f"An error occurred while fetching reviews: {str(e)}"
        msg_history.append(
            {"role": "system", "content": f"MOVIE REVIEW CONTEXT:\n\n{reviews}"}
        )


@observe
@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    await add_review_context_if_needed(message_history)

    response_message = await generate_response(client, message_history, gen_kwargs)
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

    while True:
        if "get_now_playing_movies()" in response_message.content:
            now_playing_movies = get_now_playing_movies()
            message_history.append(
                {
                    "role": "system",
                    "content": f"Current movies:\n\n {now_playing_movies}",
                }
            )

            # Stream another response with the updated message history
            response_message = await generate_response(
                client, message_history, gen_kwargs
            )
            message_history.append(
                {"role": "assistant", "content": response_message.content}
            )
            cl.user_session.set("message_history", message_history)
        elif "get_showtimes(" in response_message.content:
            movie, location = parse_function_call_args(response_message.content)
            print(f"Movie: {movie}, Location: {location}")
            try:
                showtimes = get_showtimes(movie, location)
            except Exception as e:
                showtimes = f"An error occurred while fetching showtimes: {str(e)}"

            message_history.append({"role": "system", "content": showtimes})

            # Stream another response with the updated message history
            response_message = await generate_response(
                client, message_history, gen_kwargs
            )
            message_history.append(
                {"role": "assistant", "content": response_message.content}
            )
            cl.user_session.set("message_history", message_history)
        elif "confirm_ticket_purchase(" in response_message.content:
            message_history.append(
                {
                    "role": "system",
                    "content": f"The user has confirmed their purchase for {response_message.content}",
                }
            )
            # Stream another response with the updated message history
            response_message = await generate_response(
                client, message_history, gen_kwargs
            )
            message_history.append(
                {"role": "assistant", "content": response_message.content}
            )
            cl.user_session.set("message_history", message_history)
        elif "buy_ticket(" in response_message.content:
            theater, movie, showtime = parse_function_call_args(
                response_message.content
            )
            print(f"Theater: {theater}, Movie: {movie}, Showtime: {showtime}")
            try:
                ticket_purchase_confirmation = buy_ticket(theater, movie, showtime)
            except Exception as e:
                ticket_purchase_confirmation = (
                    f"An error occurred while purchasing the ticket: {str(e)}"
                )
            message_history.append(
                {
                    "role": "system",
                    "content": f"{ticket_purchase_confirmation}",
                }
            )

            # Stream another response with the updated message history
            response_message = await generate_response(
                client, message_history, gen_kwargs
            )
            message_history.append(
                {"role": "assistant", "content": response_message.content}
            )
            cl.user_session.set("message_history", message_history)
        else:
            break
        print("Looping...")


if __name__ == "__main__":
    cl.main()
