# Use an official Python runtime as a parent image, specifically an Alpine-based image
FROM python:3.11.4-alpine

# Set the working directory in the container
WORKDIR /app

# Install Cython, GCC, and other dependencies required to compile Cython code
# Include Python3-dev and other necessary libraries like musl-dev
RUN apk add --no-cache gcc musl-dev python3-dev libffi-dev openssl-dev

# Also, install Cython and numpy (if needed) here to ensure they're available for the build
RUN pip install --no-cache-dir cython numpy

# Copy the entire current directory contents into the container at /app
COPY . /app

# Copy the service account key file into the container
COPY gifted-veld-421911-6f3cb483c45a.json /app/service-account-file.json

# Set the environment variable for Google Cloud authentication
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-file.json"
ENV PYTHONUNBUFFERED=1

# Install any needed packages specified in requirements.txt
# Adjust the path to your requirements.txt if it's not in the root of the project
RUN pip install --no-cache-dir -r requirements.txt

# Compile Cython code. 
RUN python /app/particle_filter/algo/trajectory/setup.py build_ext --inplace

RUN ls -la /app/particle_filter/algo/trajectory/
RUN mv /app/trajectory_list*.so /app/particle_filter/algo/trajectory/ 2>/dev/null || :

# Specify the command to run your application
CMD ["python", "-u", "/app/particle_filter/run_benchmark.py"]


# to build: docker build -t vix .