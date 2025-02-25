FROM continuumio/miniconda3:latest
WORKDIR /app

COPY environment.yml /app/environment.yml

# Create a Conda environment named 'ml_env' and activate it
RUN conda env create -f environment.yml && conda clean --all -y

# Ensure the environment is activated by default
SHELL ["conda", "run", "-n", "synthetic-ml", "/bin/bash", "-c"]

# Create models directory
RUN mkdir -p /app/models

# Copy the required scripts and model files into the container
COPY scripts/prediction.py /app/
COPY models/xgboost_model.pkl  /app/models/

# Expose port 9696 for the application
EXPOSE 9696

# Use Conda environment when running Gunicorn
ENTRYPOINT ["conda", "run", "-n", "ml_env", "gunicorn", "--bind", "0.0.0.0:9696", "flask_pre:app"]