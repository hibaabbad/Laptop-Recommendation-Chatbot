# Laptop Recommendation Chatbot

## Project Overview

This project is a **Laptop Recommendation Chatbot** that assists users in finding suitable laptops based on their specific requirements. It leverages machine learning and natural language processing techniques to provide personalized recommendations. The chatbot can be tested in a Jupyter Notebook environment and is also accessible through a user-friendly Streamlit interface.

## Features

- **Interactive Chatbot**: Users can ask questions about laptop specifications and receive recommendations.
- **Streamlit Interface**: A web-based interface for easy interaction with the chatbot.
- **Document Loading**: The chatbot loads laptop data from a CSV file, which serves as a database for recommendations.
- **History-Aware Retrieval**: The chatbot remembers previous interactions to provide context-aware responses.

## Prerequisites

To run this project, ensure you have the following:

- Python 3.7 or later
- A valid **GROQ API key** (sign up at [GROQ](https://console.groq.com/keys) to obtain one).

## Installation

1. Clone the repository:

   git clone https://github.com/hibaabbad/Laptop-Recommendation-Chatbot.git
   cd Laptop-Recommendation-Chatbot

2. Create a virtual environment and activate it:

    python -m venv env

    On Windows:
        .\env\Scripts\activate

    On macOS/Linux:
        source env/bin/activate

3. Install the required dependencies:

    pip install -r requirements.txt

4. Run the Chatbot:

    To launch the Streamlit interface, run the following command:
        streamlit run chatbot.py

5. Testing in Jupyter Notebook:

    Open chatbot.ipynb in Jupyter and run the cells to test the chatbot's functionality.

