# FROM python:3.9
# RUN useradd -m -u 1000 user
# USER user
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH
# WORKDIR $HOME/app
# COPY --chown=user . $HOME/app
# COPY ./requirements.txt ~/app/requirements.txt
# RUN pip install -r requirements.txt
# COPY . .
# CMD ["chainlit", "run", "app.py", "--port", "7860"]

# Use an official Python runtime as a parent image
FROM python:3.10

# Create a non-root user and switch to it for security
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory in the container
WORKDIR $HOME/app

# Copy only the requirements first to leverage Docker cache
COPY --chown=user requirements.txt $HOME/app/

# Install any needed packages specified in requirements.txt
RUN pip install --user -r requirements.txt

# Copy the rest of the application's code
COPY --chown=user . $HOME/app

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define the command to run the app
CMD ["chainlit", "run", "app.py", "--port", "7860"]

