"""
Package: service
Package for the application models and service routes
This module creates and configures the Flask app and sets up the logging
"""
import os
import sys
import logging
from flask import Flask

# Create Flask application
app = Flask(__name__)
app.config.from_object("config")

# Import the rutes After the Flask app is created
from code.service import routes, models, error_handlers

# Set up logging for production
print("Setting up logging for {}...".format(__name__))
app.logger.propagate = False
if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    # Make all log formats consistent
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s", "%Y-%m-%d %H:%M:%S %z"
    )
    for handler in app.logger.handlers:
        handler.setFormatter(formatter)
    app.logger.info("Logging handler established")

app.logger.info(70 * "*")
app.logger.info("  A M R   V E R B N E T   S E R V I C E  ".center(70, "*"))
app.logger.info(70 * "*")

app.logger.info("Service inititalized!")

