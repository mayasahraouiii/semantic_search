# Use the official Python image from the Docker Hub
FROM python:3.10-bullseye

# Set the working directory
WORKDIR /code
 
COPY ./streamlit_src /code/streamlit_src 
COPY ./requirements.txt /code/requirements.txt

RUN pip install -r  /code/requirements.txt

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_src/app.py","--server.port=80"]