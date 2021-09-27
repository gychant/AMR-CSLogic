
"""
AMR-VerbNet Service

Paths:
------
GET /amr_parsing - Returns AMR parsing results
GET /verbnet_semantics - Returns grounding results using VerbNet semantics
"""

import os
import sys
import logging
from flask import Flask, jsonify, request, url_for, make_response, abort
from werkzeug.exceptions import NotFound
# Import Flask application
from . import app
# HTTP Status Codes
from . import status

from amr_verbnet_semantics.service.amr import parse_text
from amr_verbnet_semantics.core.amr_verbnet_enhance import ground_text_to_verbnet

######################################################################
# GET INDEX
######################################################################
@app.route("/")
def index():
    """Root URL response"""
    app.logger.info("Request for Root URL")
    return (
        jsonify(
            name="AMR-VerbNet REST API Service",
            version="1.0"
        ),
        status.HTTP_200_OK,
    )


######################################################################
# AMR parsing
######################################################################
@app.route("/amr_parsing", methods=["GET"])
def amr_parsing():
    """Returns AMR parse of input text"""
    app.logger.info("AMR parsing of text ...")
    text = request.args.get("text")
    if len(text.strip()) == 0:
        make_response(jsonify({
            "error": "Empty input text."
        }), status.HTTP_200_OK)

    parse = parse_text(text)
    results = {
        "result": parse[0]
    }
    app.logger.info("Finished parsing ...")
    return make_response(jsonify(results), status.HTTP_200_OK)


######################################################################
# Ground text using VerbNet semantics
######################################################################
@app.route("/verbnet_semantics", methods=["GET"])
def verbnet_semantics():
    """Returns AMR parse of input text"""
    app.logger.info("AMR parsing of text ...")
    text = request.args.get("text")
    if len(text.strip()) == 0:
        make_response(jsonify({
            "error": "Empty input text."
        }), status.HTTP_200_OK)

    parse = ground_text_to_verbnet(text, verbose=False)
    results = {
        "text": text,
        "coreference": parse["coreference"],
        "amr_parse": parse["sentence_parses"]
    }
    app.logger.info("Finished parsing ...")
    return make_response(jsonify(results), status.HTTP_200_OK)


######################################################################
#  U T I L I T Y   F U N C T I O N S
######################################################################
def check_content_type(media_type):
    """Checks that the media type is correct"""
    content_type = request.headers.get("Content-Type")
    if content_type and content_type == media_type:
        return
    app.logger.error("Invalid Content-Type: %s", content_type)
    abort(
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        "Content-Type must be {}".format(media_type),
    )

